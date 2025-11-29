import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#extension.py

def gem(x, p=3, eps=1e-6):
    """GEM 池化"""
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class LabAlignmentModule(nn.Module):
    """
    Lab亮度对齐模块（极简设计）
    核心原理：
    1. 从特征图提取空间注意力（模拟亮度分布）
    2. 强制可见光和红外的空间注意力一致
    物理依据：
    - 可见光的亮度 ≈ Lab-L通道
    - 红外的灰度 ≈ 亮度
    - 因此两者的"亮度分布"应该一致
    优势：
    - 在特征图级别应用（保留空间信息）
    - 单一损失目标（无复杂组合）
    - 物理意义明确（空间亮度对齐）
    """

    def __init__(self, use_channel_weight=False):
        super(LabAlignmentModule, self).__init__()

        self.use_channel_weight = use_channel_weight

        if use_channel_weight:
            # 可学习的通道权重（让网络自动学习哪些通道代表亮度）
            self.channel_weight_v = nn.Parameter(torch.ones(2048, 1, 1))
            self.channel_weight_i = nn.Parameter(torch.ones(2048, 1, 1))

    def extract_luminance_map(self, feat_map, channel_weight=None):
        """
        从特征图提取亮度分布图
        Args:
            feat_map: [B, 2048, H, W] ResNet的layer4输出
            channel_weight: [2048, 1, 1] 可学习通道权重
        Returns:
            lum_map: [B, 1, H, W] 亮度分布图
        """
        # 方法1：加权求和（如果有通道权重）
        if channel_weight is not None:
            lum_map = (feat_map * channel_weight).sum(dim=1, keepdim=True)
        # 方法2：能量求和（默认）
        else:
            lum_map = feat_map.pow(2).sum(dim=1, keepdim=True)
        # 归一化到[0, 1]
        B, _, H, W = lum_map.shape
        lum_map = lum_map.view(B, -1)
        # lum_map = (lum_map - lum_map.min(dim=1, keepdim=True)[0]) / \
        #           (lum_map.max(dim=1, keepdim=True)[0] - lum_map.min(dim=1, keepdim=True)[0] + 1e-6)
        lum_map = lum_map.view(B, 1, H, W)

        return lum_map

    def forward(self, feat_v, feat_i, labels):
        """
        前向传播

        Args:
            feat_v: [B, 2048, H, W] 可见光特征图
            feat_i: [B, 2048, H, W] 红外特征图
            labels: [B] 身份标签

        Returns:
            lab_loss: 标量损失
        """
        # 处理维度（如果输入是GEM池化后的向量，报错）
        if feat_v.dim() == 2:
            raise ValueError(
                "Lab模块需要特征图输入 [B, C, H, W]，而非向量 [B, C]。"
                "请在GEM池化之前调用，即使用 x_sh 而非 sh_pl。"
            )

        # 提取亮度分布图
        if self.use_channel_weight:
            lum_v = self.extract_luminance_map(feat_v, self.channel_weight_v)
            lum_i = self.extract_luminance_map(feat_i, self.channel_weight_i)
        else:
            lum_v = self.extract_luminance_map(feat_v)
            lum_i = self.extract_luminance_map(feat_i)

        # 核心损失：同ID的亮度分布应该一致
        # 使用结构相似性而非简单的MSE（更符合视觉感知）
        loss = 0.0
        unique_labels = torch.unique(labels)

        for label in unique_labels:
            mask = labels == label
            if mask.sum() < 2:
                continue

            lum_v_class = lum_v[mask]  # [N, 1, H, W]
            lum_i_class = lum_i[mask]

            # 计算类内的亮度分布差异
            # 方法：两两配对的MSE
            N = lum_v_class.size(0)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        loss += F.mse_loss(lum_v_class[i], lum_i_class[j])

        # 归一化
        num_pairs = sum((labels == l).sum().item() ** 2 - (labels == l).sum().item()
                        for l in unique_labels)
        if num_pairs > 0:
            loss = loss / num_pairs
        else:
            loss = torch.tensor(0.0, device=feat_v.device, requires_grad=True)

        return loss

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
        """对齐共同邻居"""
        total_loss = 0.0
        valid_count = 0

        for i in range(len(common_neighbors)):
            common = common_neighbors[i]
            if len(common) == 0:
                continue

            feat_v_i = feat_v[i:i+1]
            feat_i_i = feat_i[i:i+1]
            feat_common_v = feat_v[common]
            feat_common_i = feat_i[common]

            feat_v_i_norm = F.normalize(feat_v_i, p=2, dim=1)
            feat_i_i_norm = F.normalize(feat_i_i, p=2, dim=1)
            feat_common_v_norm = F.normalize(feat_common_v, p=2, dim=1)
            feat_common_i_norm = F.normalize(feat_common_i, p=2, dim=1)

            sim_v = torch.mm(feat_v_i_norm, feat_common_v_norm.t()) / self.temperature
            sim_i = torch.mm(feat_i_i_norm, feat_common_i_norm.t()) / self.temperature

            loss = F.mse_loss(sim_v, sim_i)
            # prob_v = F.softmax(sim_v, dim=1)
            # prob_i = F.softmax(sim_i, dim=1)
            # loss = F.kl_div(prob_v.log(), prob_i, reduction='batchmean')

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