import torch
import torch.nn as nn
import torch.nn.functional as F

class rSCAplusplus(nn.Module):
    """
    rSCA+++: 确定性稀疏协同降噪模块（修正版）
    - 同图内K-NN确定性采样
    - 共享邻居索引确保协同矩阵可比
    - patch级降噪掩码
    """

    def __init__(self, dim=512, k=16, use_checkpoint=False):
        """
        Args:
            dim: 输入特征通道数（layer3输出：1024）
            k: K-NN邻居数（推荐8-16）
            use_checkpoint: 是否启用梯度检查点（节省显存）
        """
        super().__init__()
        self.k = k
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # 协同门控单元（CGC）
        self.cgc = nn.Sequential(
            nn.Linear(k, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 可选：可学习的温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, F_v, F_i):
        """
        Args:
            F_v: [B, C, H, W] VIS特征图
            F_i: [B, C, H, W] IR特征图

        Returns:
            F_v_denoised: [B, C, H, W]
            F_i_denoised: [B, C, H, W]
            loss_syn: scalar
        """
        B, C, H, W = F_v.shape
        N = H * W

        # 1. 展平特征图 [B, C, H, W] -> [B, N, C]
        F_v_flat = F_v.flatten(2).permute(0, 2, 1)  # [B, N, C]
        F_i_flat = F_i.flatten(2).permute(0, 2, 1)

        # 2. 构建共享K-NN邻居索引（关键修正）
        with torch.no_grad():  # 索引构建不需要梯度
            idx_shared = self._build_knn_index(F_v_flat, self.k)

        # 3. 计算协同分数（共享索引）
        C_v = self._compute_coherence(F_v_flat, idx_shared)
        C_i = self._compute_coherence(F_i_flat, idx_shared)

        # 4. 生成降噪掩码
        mask_v = self.cgc(C_v)  # [B, N, 1]
        mask_i = self.cgc(C_i)

        # 5. 应用掩码（patch级抑制）
        F_v_denoised = F_v_flat * mask_v
        F_i_denoised = F_i_flat * mask_i

        # 6. 恢复特征图形状 [B, N, C] -> [B, C, H, W]
        F_v_denoised = F_v_denoised.permute(0, 2, 1).reshape(B, C, H, W)
        F_i_denoised = F_i_denoised.permute(0, 2, 1).reshape(B, C, H, W)

        # 7. 协同结构对齐损失
        loss_syn = self._coherence_align_loss(C_v, C_i)

        return F_v_denoised, F_i_denoised, loss_syn

    def _build_knn_index(self, F, k):
        """
        构建同图内K-NN索引（确定性，无随机性）
        分块计算避免OOM
        """
        B, N, C = F.shape
        device = F.device

        # L2归一化
        F_norm = F / (F.norm(dim=-1, keepdim=True) + 1e-6)

        # 分块计算距离矩阵（避免N²内存爆炸）
        chunk_size = 256  # 每次处理256个patch
        indices_list = []

        for b in range(B):
            f = F_norm[b]  # [N, C]
            batch_indices = []

            for i in range(0, N, chunk_size):
                # 计算当前块与所有patch的距离
                f_chunk = f[i:i + chunk_size]  # [chunk, C]
                dist = 1 - torch.mm(f_chunk, f.T)  # [chunk, N] 余弦距离

                # 排除自身：对角线设为无穷大
                for j in range(dist.size(0)):
                    dist[j, i + j] = float('inf')

                # Top-K最近邻
                _, idx = dist.topk(k, dim=1, largest=False)  # [chunk, k]
                batch_indices.append(idx)

            indices_list.append(torch.cat(batch_indices, dim=0))  # [N, k]

        return torch.stack(indices_list)  # [B, N, k]

    def _compute_coherence(self, F, idx):
        """
        根据索引计算协同分数

        ***修正：使用高级索引替代 expand + gather 避免显存爆炸***
        """
        B, N, C = F.shape
        k = idx.size(-1)

        # L2归一化
        F_norm = F / (F.norm(dim=-1, keepdim=True) + 1e-6)  # [B, N, C]

        # --------------------- 显存安全修正区域 ---------------------
        # 1. 创建 batch 索引 [B, 1, 1] -> [B, N, k]
        # 用于索引 F_norm 的第一个维度 (Batch)
        batch_indices = torch.arange(B, device=F.device).view(B, 1, 1).expand(-1, N, k)

        # 2. 使用高级索引 F_norm[B_idx, N_idx, C_idx] 获取邻居特征 [B, N, k, C]
        # F_norm: [B, N, C]
        # batch_indices: [B, N, k] (用于索引 B 维)
        # idx: [B, N, k] (用于索引 N 维)
        # 结果 F_neighbors: [B, N, k, C]
        F_neighbors = F_norm[batch_indices, idx, :]

        # --------------------- 显存安全修正区域 ---------------------

        # 计算余弦相似度（协同分数）
        # F_norm.unsqueeze(2): [B, N, 1, C]
        C = (F_norm.unsqueeze(2) * F_neighbors).sum(dim=-1)  # [B, N, k]
        C = C / self.temperature  # 可学习温度

        return C

    def _coherence_align_loss(self, C_v, C_i):
        """
        协同结构对齐损失（归一化MSE）
        """
        # 行级L2归一化
        C_v_norm = F.normalize(C_v, p=2, dim=-1)
        C_i_norm = F.normalize(C_i, p=2, dim=-1)

        # MSE损失
        loss = F.mse_loss(C_v_norm, C_i_norm)
        loss = 100 * loss
        return loss

