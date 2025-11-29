import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#extension.py

def gem(x, p=3, eps=1e-6):
    """GEM 池化"""
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class LabAlignmentModule(nn.Module):
    def __init__(self, use_channel_weight=True, pool_size=4):
        """
        Args:
            use_channel_weight: 是否使用通道加权
            pool_size: 池化核大小，建议2-4（降低空间分辨率，减少计算量）
        """
        super().__init__()
        self.use_channel_weight = use_channel_weight
        self.pool_size = pool_size

        # 保留独立通道权重（保持表达能力）
        if use_channel_weight:
            self.channel_weight_v = nn.Parameter(torch.ones(2048, 1, 1)*0.5)
            self.channel_weight_i = nn.Parameter(torch.ones(2048, 1, 1)*0.5)

    def extract_luminance_map(self, feat_map, channel_weight=None):
        """提取亮度图（优化版）"""
        # 【优化1】先池化降低空间分辨率
        if self.pool_size > 1:
            feat_map = F.adaptive_avg_pool2d(feat_map,
                                             (feat_map.size(2) // self.pool_size,
                                              feat_map.size(3) // self.pool_size))

        if channel_weight is not None:
            # 保留原始加权逻辑
            norm_val = channel_weight.norm(dim=0, keepdim=True).norm(dim=2, keepdim=True)
            w = channel_weight / (norm_val + 1e-6)
            lum_map = (feat_map * w).sum(dim=1, keepdim=True)
        else:
            lum_map = feat_map.pow(2).sum(dim=1, keepdim=True).sqrt()

        # Min-Max 归一化（保持不变）
        B, _, H, W = lum_map.shape
        lum_map = lum_map.view(B, -1)
        min_val = lum_map.min(dim=1, keepdim=True)[0]
        max_val = lum_map.max(dim=1, keepdim=True)[0]
        lum_map = (lum_map - min_val) / (max_val - min_val + 1e-6)
        return lum_map.view(B, 1, H, W)

    def forward(self, feat_v, feat_i, labels):
        if feat_v.dim() == 2:
            raise ValueError("需要特征图输入 [B, C, H, W]")

        # 提取亮度图
        if self.use_channel_weight:
            lum_v = self.extract_luminance_map(feat_v, self.channel_weight_v)
            lum_i = self.extract_luminance_map(feat_i, self.channel_weight_i)
        else:
            lum_v = self.extract_luminance_map(feat_v)
            lum_i = self.extract_luminance_map(feat_i)

        # 【优化2】矩阵化计算损失
        return self._compute_loss_vectorized(lum_v, lum_i, labels)

    def _compute_loss_vectorized(self, lum_v, lum_i, labels):
        """向量化损失计算（移除嵌套for循环）"""
        unique_labels = torch.unique(labels)

        # 存储所有同ID对的损失
        all_mse_losses = []
        all_cos_losses = []

        for label in unique_labels:
            mask = labels == label
            if mask.sum() < 2:
                continue

            lv = lum_v[mask]  # [N_v, 1, H, W]
            li = lum_i[mask]  # [N_i, 1, H, W]

            N_v, N_i = lv.size(0), li.size(0)

            # 【关键优化】扩展为 [N_v, N_i, 1, H, W] 的配对张量
            lv_expand = lv.unsqueeze(1).expand(N_v, N_i, 1, lv.size(2), lv.size(3))
            li_expand = li.unsqueeze(0).expand(N_v, N_i, 1, li.size(2), li.size(3))

            # MSE损失：[N_v, N_i]
            mse_loss = F.mse_loss(lv_expand, li_expand, reduction='none')
            mse_loss = mse_loss.mean(dim=(2, 3, 4))  # 对空间和通道维度求平均

            # Cosine损失：[N_v, N_i]
            lv_flat = lv.view(N_v, -1).unsqueeze(1)  # [N_v, 1, H*W]
            li_flat = li.view(N_i, -1).unsqueeze(0)  # [1, N_i, H*W]
            cos_sim = F.cosine_similarity(lv_flat, li_flat, dim=2)  # [N_v, N_i]
            cos_loss = 1 - cos_sim

            all_mse_losses.append(mse_loss.view(-1))
            all_cos_losses.append(cos_loss.view(-1))

        if len(all_mse_losses) == 0:
            return torch.tensor(0.0, device=lum_v.device, requires_grad=True)

        # 拼接所有损失并求平均
        total_mse = torch.cat(all_mse_losses).mean()
        total_cos = torch.cat(all_cos_losses).mean()

        return 0.1 * total_mse + total_cos


class GraphContrastiveAlignment(nn.Module):
    """
    GCA (Graph Contrastive Alignment) - 图对比对齐
    创新1: 双层邻居系统（共同邻居 vs 特异邻居）
    创新2: 模态特异性保护器（对抗式保护）
    修复: 确保损失为正值
    """

    def __init__(self, k_neighbors=8, temperature=0.15, common_ratio=0.6):
        super(GraphContrastiveAlignment, self).__init__()
        self.k = k_neighbors
        self.temperature = temperature
        self.common_ratio = common_ratio
        # 模态特异性判别器
        self.modality_discriminator = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

        self.NEG_INF = -65000.0

    def build_dual_neighbor_graph(self, feat_v, feat_i, labels):
        """
        双层邻居系统（核心创新）
        返回：共同邻居、V特异邻居、I特异邻居
        """
        if feat_v.dim() > 2:
            feat_v = gem(feat_v).squeeze()
            feat_v = feat_v.view(feat_v.size(0), -1)
        if feat_i.dim() > 2:
            feat_i = gem(feat_i).squeeze()
            feat_i = feat_i.view(feat_i.size(0), -1)

        B = feat_v.size(0)

        feat_v_norm = F.normalize(feat_v.float(), p=2, dim=1)
        feat_i_norm = F.normalize(feat_i.float(), p=2, dim=1)

        sim_v = torch.mm(feat_v_norm, feat_v_norm.t())
        sim_i = torch.mm(feat_i_norm, feat_i_norm.t())

        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        label_mask.fill_diagonal_(False)

        same_class_counts = label_mask.sum(dim=1)
        if same_class_counts.max() == 0:
            return None, None, None

        common_neighbors = []
        specific_v_neighbors = []
        specific_i_neighbors = []

        k_common = max(1, int(self.k * self.common_ratio))
        k_specific = self.k - k_common

        for i in range(B):
            num_same_class = same_class_counts[i].item()
            if num_same_class == 0:
                print(f"[GCA-WARN] 样本{i}（标签{labels[i].item()}）无同身份样本，跳过邻居筛选")
                common_neighbors.append(torch.tensor([], dtype=torch.long, device=feat_v.device))
                specific_v_neighbors.append(torch.tensor([], dtype=torch.long, device=feat_v.device))
                specific_i_neighbors.append(torch.tensor([], dtype=torch.long, device=feat_v.device))
                continue

            # V模态邻居
            sim_v_i = sim_v[i].clone()
            sim_v_i[~label_mask[i]] = self.NEG_INF
            k_v = min(self.k * 2, num_same_class)
            _, neighbors_v = torch.topk(sim_v_i, k_v)

            # I模态邻居
            sim_i_i = sim_i[i].clone()
            sim_i_i[~label_mask[i]] = self.NEG_INF
            k_i = min(self.k * 2, num_same_class)
            _, neighbors_i = torch.topk(sim_i_i, k_i)

            # 计算交集和差集
            neighbors_v_set = set(neighbors_v.cpu().numpy())
            neighbors_i_set = set(neighbors_i.cpu().numpy())

            common = neighbors_v_set & neighbors_i_set
            specific_v = neighbors_v_set - common
            specific_i = neighbors_i_set - common

            # 限制数量
            common = torch.tensor(list(common)[:k_common], dtype=torch.long, device=feat_v.device)
            specific_v = torch.tensor(list(specific_v)[:k_specific], dtype=torch.long, device=feat_v.device)
            specific_i = torch.tensor(list(specific_i)[:k_specific], dtype=torch.long, device=feat_v.device)

            common_neighbors.append(common)
            specific_v_neighbors.append(specific_v)
            specific_i_neighbors.append(specific_i)

        return common_neighbors, specific_v_neighbors, specific_i_neighbors

    def alignment_loss(self, feat_v, feat_i, common_neighbors):
        """对齐共同邻居：修复为跨模态对齐"""
        total_loss = 0.0
        valid_count = 0

        for i in range(len(common_neighbors)):
            common = common_neighbors[i]
            if len(common) == 0:
                continue

            feat_v_i = feat_v[i:i + 1]
            feat_i_i = feat_i[i:i + 1]

            feat_common_v = feat_v[common]
            feat_common_i = feat_i[common]

            feat_v_i_norm = F.normalize(feat_v_i, p=2, dim=1)
            feat_i_i_norm = F.normalize(feat_i_i, p=2, dim=1)
            feat_common_v_norm = F.normalize(feat_common_v, p=2, dim=1)
            feat_common_i_norm = F.normalize(feat_common_i, p=2, dim=1)

            # --- 关键修改部分：实现跨模态对齐 ---

            # 1. V 样本在 I 空间中看到的邻居分布 (V-to-I)
            # 使用 V 查询样本 (feat_v_i_norm) 对比 I 模态邻居 (feat_common_i_norm)
            sim_v_to_i = torch.mm(feat_v_i_norm, feat_common_i_norm.t())

            # 2. I 样本在 V 空间中看到的邻居分布 (I-to-V)
            # 使用 I 查询样本 (feat_i_i_norm) 对比 V 模态邻居 (feat_common_v_norm)
            sim_i_to_v = torch.mm(feat_i_i_norm, feat_common_v_norm.t())

            # if i == 0 or i==1:
            #     print(f"[DEBUG] sim_v_to_i range {i}: [{sim_v_to_i.min():.3f}, {sim_v_to_i.max():.3f}]")
            #     print(f"[DEBUG] sim_i_to_v range {i}: [{sim_i_to_v.min():.3f}, {sim_i_to_v.max():.3f}]")
            # 目标：强制 Sim(V, I) 和 Sim(I, V) 对齐
            # loss = F.mse_loss(sim_v_to_i, sim_i_to_v)

            # 将 [1, N] 展平为 [N]
            sim_v_to_i = sim_v_to_i.squeeze(0)
            sim_i_to_v = sim_i_to_v.squeeze(0)


            # 计算余弦相似度（值越大越好），损失取 1 - cos(theta)（值越小越好）
            cos_sim = F.cosine_similarity(sim_v_to_i, sim_i_to_v, dim=0)

            # 损失： 1 - Cosine Similarity
            loss = 1.0 - cos_sim
            total_loss += loss
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=feat_v.device)

        return total_loss / valid_count

    def diversity_preservation_loss(self, feat_v, feat_i, specific_v, specific_i):
        """
        保护特异邻居（核心创新 - 修复版）
        修复：改用正向约束，不用负号
        """
        all_specific_v = []
        all_specific_i = []

        for i in range(len(specific_v)):
            if len(specific_v[i]) > 0:
                all_specific_v.append(feat_v[specific_v[i]])
            if len(specific_i[i]) > 0:
                all_specific_i.append(feat_i[specific_i[i]])

        if len(all_specific_v) == 0 or len(all_specific_i) == 0:
            return torch.tensor(0.0, device=feat_v.device)

        specific_v_feats = torch.cat(all_specific_v, dim=0)
        specific_i_feats = torch.cat(all_specific_i, dim=0)

        # 判别器预测
        logits_v = self.modality_discriminator(specific_v_feats)
        logits_i = self.modality_discriminator(specific_i_feats)

        labels_v = torch.zeros(len(specific_v_feats), dtype=torch.long, device=feat_v.device)
        labels_i = torch.ones(len(specific_i_feats), dtype=torch.long, device=feat_i.device)

        # 修复：判别器能区分 = 损失低（正向）
        # 我们希望判别器能区分 → 最小化交叉熵（而非最大化）
        div_loss = F.cross_entropy(logits_v, labels_v) + F.cross_entropy(logits_i, labels_i)

        return div_loss  # 小权重，辅助损失

    def forward(self, feat_v, feat_i, labels, is_shared=True):
        """前向传播"""
        common, spec_v, spec_i = self.build_dual_neighbor_graph(feat_v, feat_i, labels)

        gca_loss = 0.0

        if common is None:
            print("[INFO] common is None")
            return torch.tensor(0.0, device=feat_v.device, requires_grad=True)

        if is_shared:
            # 1. 对齐共同邻居
            align_loss = self.alignment_loss(feat_v, feat_i, common)
            gca_loss += align_loss

        else:
            # 2. 保护特异邻居（判别器能区分 = 损失低）
            div_loss = self.diversity_preservation_loss(feat_v, feat_i, spec_v, spec_i)
            gca_loss += div_loss

        if torch.isnan(gca_loss) or torch.isinf(gca_loss):
            gca_loss = torch.tensor(0.0, device=feat_v.device, requires_grad=True)

        # 安全检查
        gca_loss = torch.clamp(gca_loss, min=0.0)

        return gca_loss

