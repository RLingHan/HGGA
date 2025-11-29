import torch
import torch.nn as nn
import math


class RGB_Lab_Converter:
    """
    RGB ↔ Lab 色彩空间转换器（优化版）

    改进：
    1. 缓存转换矩阵（避免重复.to(device)）
    2. 全部使用 PyTorch 操作
    """

    def __init__(self):
        # 转换矩阵（固定值）
        self.rgb_to_xyz_matrix = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=torch.float32)

        # 参考白点 D65
        self.ref_white = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32)

        # 缓存逆矩阵（避免重复计算）
        self.xyz_to_rgb_matrix = torch.inverse(self.rgb_to_xyz_matrix.t())

        # 设备缓存
        self._cached_device = None

    def _ensure_device(self, device):
        """确保矩阵在正确的设备上（只转移一次）"""
        if self._cached_device != device:
            self.rgb_to_xyz_matrix = self.rgb_to_xyz_matrix.to(device)
            self.xyz_to_rgb_matrix = self.xyz_to_rgb_matrix.to(device)
            self.ref_white = self.ref_white.to(device)
            self._cached_device = device

    def rgb_to_lab(self, rgb):
        """
        Args:
            rgb: [C, H, W] 或 [B, C, H, W], 范围 [0, 1]
        Returns:
            lab: 同shape, L∈[0,100], a/b∈[-128,127]
        """
        single_image = False
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
            single_image = True

        device = rgb.device
        self._ensure_device(device)  # ✅ 优化：缓存设备转换

        # Step 1: RGB → XYZ
        rgb_linear = self._linearize_rgb(rgb)

        B, C, H, W = rgb_linear.shape
        rgb_flat = rgb_linear.permute(0, 2, 3, 1).reshape(-1, 3)
        xyz_flat = torch.mm(rgb_flat, self.rgb_to_xyz_matrix.t())
        xyz = xyz_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        xyz = xyz / self.ref_white.view(1, 3, 1, 1)

        # Step 2: XYZ → Lab
        f_xyz = self._f_transform(xyz)

        L = 116 * f_xyz[:, 1] - 16
        a = 500 * (f_xyz[:, 0] - f_xyz[:, 1])
        b = 200 * (f_xyz[:, 1] - f_xyz[:, 2])

        lab = torch.stack([L, a, b], dim=1)

        if single_image:
            lab = lab.squeeze(0)

        return lab

    def lab_to_rgb(self, lab):
        """
        Args:
            lab: [C, H, W] 或 [B, C, H, W]
        Returns:
            rgb: 同shape, 范围 [0, 1]
        """
        single_image = False
        if lab.dim() == 3:
            lab = lab.unsqueeze(0)
            single_image = True

        device = lab.device
        self._ensure_device(device)

        L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

        # Lab → XYZ
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200

        xyz = torch.stack([
            self._f_inverse(fx),
            self._f_inverse(fy),
            self._f_inverse(fz)
        ], dim=1)

        xyz = xyz * self.ref_white.view(1, 3, 1, 1)

        # XYZ → RGB
        B, C, H, W = xyz.shape
        xyz_flat = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
        rgb_linear_flat = torch.mm(xyz_flat, self.xyz_to_rgb_matrix.t())  # ✅ 使用缓存的逆矩阵
        rgb_linear = rgb_linear_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        rgb = self._delinearize_rgb(rgb_linear)
        rgb = torch.clamp(rgb, 0, 1)

        if single_image:
            rgb = rgb.squeeze(0)

        return rgb

    # ========== 辅助函数 ==========

    def _linearize_rgb(self, rgb):
        """移除 gamma 校正"""
        mask = rgb > 0.04045
        linear = torch.where(
            mask,
            torch.pow((rgb + 0.055) / 1.055, 2.4),
            rgb / 12.92
        )
        return linear

    def _delinearize_rgb(self, rgb_linear):
        """添加 gamma 校正"""
        mask = rgb_linear > 0.0031308
        rgb = torch.where(
            mask,
            1.055 * torch.pow(rgb_linear, 1 / 2.4) - 0.055,
            12.92 * rgb_linear
        )
        return rgb

    def _f_transform(self, t):
        """Lab 的非线性变换"""
        delta = 6 / 29
        threshold = delta ** 3
        mask = t > threshold
        result = torch.where(
            mask,
            torch.pow(t, 1 / 3),
            (t / (3 * delta ** 2)) + (4 / 29)
        )
        return result

    def _f_inverse(self, t):
        """Lab 的反向变换"""
        delta = 6 / 29
        mask = t > delta
        result = torch.where(
            mask,
            torch.pow(t, 3),
            3 * delta ** 2 * (t - 4 / 29)
        )
        return result


class RandomColoring_Lab(object):
    """
    基于 Lab 的数据增强（全 PyTorch 优化版）

    改进：
    1. 全部使用 torch.rand / torch.randint（避免 Numpy）
    2. 改进模态判断逻辑
    3. 向量化操作（减少循环）
    """
    # ✅ 单例模式：共享 converter
    _converter = None

    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.33):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

        if RandomColoring_Lab._converter is None:
            RandomColoring_Lab._converter = RGB_Lab_Converter()
        self.converter = RandomColoring_Lab._converter

    def __call__(self, img):
        """
        Args:
            img: PIL Image 或 Tensor [C, H, W]
        Returns:
            img: Tensor [C, H, W]
        """
        # PIL → Tensor
        if not isinstance(img, torch.Tensor):
            import torchvision.transforms.functional as F
            img = F.to_tensor(img)

        # ✅ 修复1: 使用 PyTorch 随机数
        if torch.rand(1).item() >= self.p:
            return img
        device = img.device
        # ✅ 修复2: 改进的模态判断
        is_rgb = self._detect_modality(img)
        # RGB → Lab
        img_lab = self.converter.rgb_to_lab(img)
        # 归一化到 [0, 1]
        L = img_lab[0] / 100.0
        a = (img_lab[1] + 128) / 255.0
        b = (img_lab[2] + 128) / 255.0
        # ===== 5次随机patch增强 =====
        for _ in range(5):
            patch_info = self._get_random_patch_torch(img.shape[1:], device)
            if patch_info is None:
                continue
            x1, y1, h_size, w_size = patch_info
            # ✅ 修复3: 全部使用 PyTorch 随机数
            if is_rgb == 1:  # 可见光
                # L通道：提升亮度
                rand_val = 0.6 + 0.4 * torch.rand(1, device=device).item()
                L[x1:x1 + h_size, y1:y1 + w_size] = (
                        L[x1:x1 + h_size, y1:y1 + w_size] * 0.9 + 0.1 * rand_val
                )
                # a/b通道：中性灰
                a[x1:x1 + h_size, y1:y1 + w_size] = (
                        a[x1:x1 + h_size, y1:y1 + w_size] * 0.5 + 0.5 * 0.5
                )
                b[x1:x1 + h_size, y1:y1 + w_size] = (
                        b[x1:x1 + h_size, y1:y1 + w_size] * 0.5 + 0.5 * 0.5
                )
            else:  # 红外
                # L通道：平衡
                rand_val = torch.rand(1, device=device).item()
                L[x1:x1 + h_size, y1:y1 + w_size] = (
                        L[x1:x1 + h_size, y1:y1 + w_size] * 0.5 + 0.5 * rand_val
                )

                # a/b通道：补偿
                rand_a = 0.4 + 0.2 * torch.rand(1, device=device).item()
                rand_b = 0.4 + 0.2 * torch.rand(1, device=device).item()

                a[x1:x1 + h_size, y1:y1 + w_size] = (
                        a[x1:x1 + h_size, y1:y1 + w_size] * 0.5 + 0.5 * rand_a
                )
                b[x1:x1 + h_size, y1:y1 + w_size] = (
                        b[x1:x1 + h_size, y1:y1 + w_size] * 0.5 + 0.5 * rand_b
                )
        # 反归一化
        L = L * 100.0
        a = a * 255.0 - 128
        b = b * 255.0 - 128
        # Lab → RGB
        img_lab_new = torch.stack([L, a, b], dim=0)
        img_rgb = self.converter.lab_to_rgb(img_lab_new)
        return img_rgb

    def _detect_modality(self, img):
        """
        改进的模态判断
        红外特征：
        1. 三通道高度相似（标准差小）
        2. 对比度低（直方图集中）
        """
        # 方法1: 通道相似度
        channel_std = torch.std(img, dim=0).mean().item()
        # 方法2: 对比度
        img_std = img.std().item()
        # 组合判断
        if channel_std < 0.01 and img_std < 0.15:
            return 0  # 红外
        else:
            return 1  # 可见光

    def _get_random_patch_torch(self, img_size, device):
        """
        ✅ 全 PyTorch 实现的随机patch生成
        Args:
            img_size: (H, W)
            device: torch.device
        Returns:
            (x1, y1, h_size, w_size) 或 None
        """
        H, W = img_size
        area = H * W
        # ✅ 使用 PyTorch 随机数
        target_area = (self.sl + (self.sh - self.sl) * torch.rand(1, device=device).item()) * area
        aspect_ratio = self.r1 + (self.r2 - self.r1) * torch.rand(1, device=device).item()
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        if w < W and h < H:
            # ✅ 使用 torch.randint
            x1 = torch.randint(0, H - h, (1,), device=device).item()
            y1 = torch.randint(0, W - w, (1,), device=device).item()
            return x1, y1, h, w

        return None

