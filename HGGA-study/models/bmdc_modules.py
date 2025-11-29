import torch
import torch.nn as nn
import torch.nn.functional as F


class MHFEM(nn.Module):
    """
    Modality-aware High-Frequency Enhancement Module
    """
    def __init__(self, dim, r=16):
        super(MHFEM, self).__init__()
        # 双路径高频提取
        self.texture_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        laplacian = torch.tensor([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        with torch.no_grad():
            self.texture_conv.weight.data = laplacian.repeat(dim, 1, 1, 1)

        self.texture_conv.weight.requires_grad = True

        self.edge_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel = (sobel_x.abs() + sobel_y.abs()) / 2
        with torch.no_grad():
            self.edge_conv.weight.data = sobel.repeat(dim, 1, 1, 1)
        self.edge_conv.weight.requires_grad = True
        # 模态自适应权重
        self.modality_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim * 2, 1),
        )
        # 全局增强强度
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_weights=False):
        """
        Args:
            x: [B, C, H, W]
            return_weights: 是否返回权重(用于可视化)
        """
        B, C, H, W = x.shape
        # 提取高频
        x_texture = self.texture_conv(x)
        x_edge = self.edge_conv(x)
        # 模态自适应权重
        modality_weights = self.modality_gate(x)
        w_texture = torch.sigmoid(modality_weights[:, :C, :, :])
        w_edge = torch.sigmoid(modality_weights[:, C:, :, :])
        # 归一化
        w_sum = w_texture + w_edge + 1e-6
        w_texture = w_texture / w_sum
        w_edge = w_edge / w_sum
        # 融合
        x_hf = w_texture * x_texture + w_edge * x_edge
        # 全局门控
        alpha = self.global_gate(x)
        out = x + alpha * x_hf
        if return_weights:
            stats = {
                'w_texture': w_texture.mean().item(),
                'w_edge': w_edge.mean().item(),
                'alpha': alpha.mean().item()
            }
            return out, stats

        return out



class CMDP_Calculator:
    @staticmethod
    def compute(sh_feat, sp_feat, sub_labels, b=0.3, tau=0.25):
        original_dtype = sh_feat.dtype
        sh_f = sh_feat.detach().float()
        sp_f = sp_feat.detach().float()

        # 归一化
        sh_f = F.normalize(sh_f, p=2, dim=1)
        sp_f = F.normalize(sp_f, p=2, dim=1)

        mask_v = (sub_labels == 0)
        mask_i = (sub_labels == 1)

        if mask_v.sum() == 0 or mask_i.sum() == 0:
            return torch.tensor(0.5, device=sh_feat.device, dtype=original_dtype)

        # ===== 改用余弦距离 =====
        center_sh_v = F.normalize(sh_f[mask_v].mean(0, keepdim=True), p=2, dim=1).squeeze()
        center_sh_i = F.normalize(sh_f[mask_i].mean(0, keepdim=True), p=2, dim=1).squeeze()
        # 余弦距离 = 1 - 余弦相似度
        d_sh = 1 - torch.dot(center_sh_v, center_sh_i)

        center_sp_v = F.normalize(sp_f[mask_v].mean(0, keepdim=True), p=2, dim=1).squeeze()
        center_sp_i = F.normalize(sp_f[mask_i].mean(0, keepdim=True), p=2, dim=1).squeeze()
        d_sp = 1 - torch.dot(center_sp_v, center_sp_i)

        d_raw = d_sh + 0.5 * d_sp  # 范围 [0, 3]

        # ===== 调整b和tau =====
        D = torch.sigmoid((d_raw - 0.3) / 0.25)  # b=0.3, tau=0.25

        return D.to(original_dtype)

class IPCMDA_Classifier(nn.Module):
    def __init__(self, dim, num_classes):
        super(IPCMDA_Classifier, self).__init__()
        self.classifier = nn.Linear(dim, num_classes, bias=False)

        # 门控网络
        mid_dim = dim // 4
        self.gate_fc = nn.Sequential(
            nn.Linear(dim * 2 + 1, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, dim),
            nn.Sigmoid()
        )
        self.delta = nn.Parameter(torch.ones(dim) * 0.1)

    def forward(self, f_sh, f_sp, D):
        # 类型适配
        if isinstance(D, float):
            D = torch.tensor(D, device=f_sh.device, dtype=f_sh.dtype)

        if D.dim() == 0:
            D = D.view(1)
        D = D.view(-1, 1).expand(f_sh.size(0), 1)

        # 拼接 BN 后的特征
        concat_feat = torch.cat([f_sh, f_sp, D], dim=1)
        g = self.gate_fc(concat_feat)

        f_final = f_sh + g * (self.delta * f_sp)
        logits = self.classifier(f_final)
        return logits