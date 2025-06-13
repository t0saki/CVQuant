"""
Low Rank Factorization ResNet implementation using SVD compression
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Tuple, Optional
from .quantizable_resnet import resnet18_quantizable, resnet50_quantizable


class LowRankFactorization:
    """SVD-based Low Rank Factorization for neural network layers"""

    def __init__(self, epsilon: float = 0.3, device: torch.device = None):
        self.epsilon = epsilon
        self.device = device or torch.device('cpu')

    def compress_layer(self, layer: nn.Module, epsilon: Optional[float] = None) -> Tuple[nn.Module, int, int]:
        """
        使用SVD压缩单个层
        Args:
            layer: 要压缩的层
            epsilon: 能量阈值，如果为None则使用默认值
        Returns:
            压缩后的层、原始参数数量、压缩后参数数量
        """
        if epsilon is None:
            epsilon = self.epsilon

        # 处理Linear层
        if isinstance(layer, nn.Linear):
            return self._compress_linear_layer(layer, epsilon)

        # 处理Conv2d层
        elif isinstance(layer, nn.Conv2d):
            return self._compress_conv2d_layer(layer, epsilon)

        return layer, 0, 0  # 如果不是支持的层类型，返回原始层

    def _compress_linear_layer(self, layer: nn.Linear, epsilon: float) -> Tuple[nn.Module, int, int]:
        """压缩Linear层"""
        W = layer.weight.data.cpu()

        # 运行SVD
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # 找到保留指定能量的秩
        energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
        rank = torch.searchsorted(energy, 1 - epsilon).item() + 1

        # 检查分解是否能减少参数数量
        old_size = W.numel()
        new_size = rank * (W.shape[0] + W.shape[1])

        if new_size < old_size:
            # 定义低秩分解
            U_r = U[:, :rank] @ torch.diag(S[:rank])
            V_r = Vh[:rank, :]

            # 创建两个线性层替换原始层
            compressed_layer = nn.Sequential(
                nn.Linear(W.shape[1], rank, bias=False),
                nn.Linear(rank, W.shape[0], bias=True)
            )
            compressed_layer[0].weight.data = V_r.to(self.device)
            compressed_layer[1].weight.data = U_r.to(self.device)
            if layer.bias is not None:
                compressed_layer[1].bias.data = layer.bias.data.to(self.device)

            return compressed_layer, old_size, new_size

        return layer, old_size, old_size

    # def _compress_conv2d_layer(self, layer: nn.Conv2d, epsilon: float) -> Tuple[nn.Module, int, int]:
    #     """压缩Conv2d层"""
    #     W = layer.weight.data.cpu()
    #     OC, IC, kH, kW = W.shape
    #
    #     # 重塑为2D矩阵
    #     W_flat = W.view(OC, -1)
    #
    #     # 运行SVD
    #     U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
    #
    #     # 找到保留指定能量的秩
    #     energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
    #     rank = torch.searchsorted(energy, 1 - epsilon).item() + 1
    #
    #     # 检查分解是否能减少参数数量
    #     old_size = W.numel()
    #     new_size = rank * (IC * kH * kW + OC)
    #
    #     if new_size < old_size:
    #         # 定义低秩分解
    #         U_r = U[:, :rank] @ torch.diag(S[:rank])
    #         V_r = Vh[:rank, :]
    #
    #         # 创建两个卷积层替换原始层
    #         conv1 = nn.Conv2d(
    #             in_channels=IC,
    #             out_channels=rank,
    #             kernel_size=1,
    #             stride=1,
    #             padding=0,
    #             bias=False
    #         )
    #         conv2 = nn.Conv2d(
    #             in_channels=rank,
    #             out_channels=OC,
    #             kernel_size=(kH, kW),
    #             stride=layer.stride,
    #             padding=layer.padding,
    #             bias=(layer.bias is not None)
    #         )
    #
    #         conv1.weight.data = V_r.view(rank, IC, kH, kW).to(self.device)
    #         conv2.weight.data = U_r.view(OC, rank, 1, 1).to(self.device)
    #         if layer.bias is not None:
    #             conv2.bias.data = layer.bias.data.to(self.device)
    #
    #         return nn.Sequential(conv1, conv2), old_size, new_size
    #
    #     return layer, old_size, old_size

    def _compress_conv2d_layer(self, layer: nn.Conv2d, epsilon: float) -> Tuple[nn.Module, int, int]:
        """压缩Conv2d层"""
        W = layer.weight.data.cpu()
        OC, IC, kH, kW = W.shape

        # 对于 1x1 卷积，SVD 分解没有意义，直接返回
        if kH == 1 and kW == 1:
            return layer, W.numel(), W.numel()

        # 重塑为2D矩阵
        W_flat = W.view(OC, -1)
        # 运行SVD
        U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
        # 找到保留指定能量的秩
        energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
        rank = torch.searchsorted(energy, 1 - epsilon).item() + 1

        # 检查分解是否能减少参数数量
        # 新参数 = (IC * kH * kW * rank) + (rank * OC * 1 * 1)
        old_size = W.numel()
        new_size = rank * (IC * kH * kW + OC)

        if new_size < old_size:
            # 定义低秩分解的权重
            # Vh (r, IC*kH*kW) -> V_r_w (r, IC, kH, kW)
            # U (OC, r) @ S(r) -> U_r_w (OC, r, 1, 1)
            U_r = U[:, :rank]
            S_r = torch.diag(S[:rank])
            V_r = Vh[:rank, :]

            # 将奇异值 S 合并到其中一个矩阵中，通常是第一个或第二个
            # 这里我们合并到 U 中，与原始代码类似
            U_r_w = (U_r @ S_r).view(OC, rank, 1, 1)
            V_r_w = V_r.view(rank, IC, kH, kW)

            # --- 这是修正的关键部分 ---
            # 创建两个卷积层替换原始层
            # conv1: kxk 卷积，继承原始层的 stride 和 padding，但不使用偏置
            # conv2: 1x1 卷积，恢复通道数，并应用原始的偏置

            conv1 = nn.Conv2d(
                in_channels=IC,
                out_channels=rank,
                kernel_size=(kH, kW),  # <--- 修正: 定义与权重匹配 (kxk)
                stride=layer.stride,  # <--- 修正: stride 和 padding 在第一个卷积中
                padding=layer.padding,
                groups=layer.groups,
                dilation=layer.dilation,
                bias=False  # <--- 修正: 偏置移到第二个卷积
            )

            conv2 = nn.Conv2d(
                in_channels=rank,
                out_channels=OC,
                kernel_size=1,  # <--- 修正: 定义与权重匹配 (1x1)
                stride=1,
                padding=0,
                bias=(layer.bias is not None)  # <--- 修正: 在这里应用偏置
            )

            # 权重分配现在与层定义一致
            conv1.weight.data = V_r_w.to(self.device)
            conv2.weight.data = U_r_w.to(self.device)

            if layer.bias is not None:
                conv2.bias.data = layer.bias.data.to(self.device)

            return nn.Sequential(conv1, conv2), old_size, new_size

        return layer, old_size, old_size

    def compress_model(self, model: nn.Module, epsilon: Optional[float] = None) -> Tuple[nn.Module, dict]:
        """
        压缩整个模型
        Args:
            model: 要压缩的模型
            epsilon: 能量阈值
        Returns:
            压缩后的模型和统计信息
        """
        if epsilon is None:
            epsilon = self.epsilon

        compressed_model = deepcopy(model)

        total_old_size = 0
        total_new_size = 0
        compression_stats = {}

        for name, module in compressed_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 获取父模块和属性名
                if '.' in name:
                    parent, attr = name.rsplit('.', 1)
                    parent_module = compressed_model
                    for part in parent.split('.'):
                        parent_module = getattr(parent_module, part)
                else:
                    parent_module = compressed_model
                    attr = name

                # 压缩层
                new_layer, old_size, new_size = self.compress_layer(module, epsilon)
                total_old_size += old_size
                total_new_size += new_size

                # 记录统计信息
                if old_size > 0:
                    compression_stats[name] = {
                        'old_size': old_size,
                        'new_size': new_size,
                        'reduction': old_size - new_size,
                        'compression_ratio': old_size / max(new_size, 1)
                    }

                # 替换层
                setattr(parent_module, attr, new_layer)

        stats = {
            'total_old_size': total_old_size,
            'total_new_size': total_new_size,
            'total_reduction': total_old_size - total_new_size,
            'compression_ratio': total_old_size / max(total_new_size, 1),
            'epsilon': epsilon,
            'layer_stats': compression_stats
        }

        return compressed_model, stats


def resnet18_low_rank(pretrained: bool = True, num_classes: int = 1000,
                      epsilon: float = 0.3, device: torch.device = None) -> nn.Module:
    """创建经过低秩分解的ResNet-18模型"""
    # 首先获取基础的quantizable ResNet-18
    base_model = resnet18_quantizable(pretrained=pretrained, num_classes=num_classes)

    # 应用低秩分解
    low_rank_processor = LowRankFactorization(epsilon=epsilon, device=device)
    compressed_model, stats = low_rank_processor.compress_model(base_model, epsilon)

    # 将统计信息添加到模型作为属性
    compressed_model.compression_stats = stats

    return compressed_model


def resnet50_low_rank(pretrained: bool = True, num_classes: int = 1000,
                      epsilon: float = 0.3, device: torch.device = None) -> nn.Module:
    """创建经过低秩分解的ResNet-50模型"""
    # 首先获取基础的quantizable ResNet-50
    base_model = resnet50_quantizable(pretrained=pretrained, num_classes=num_classes)

    # 应用低秩分解
    low_rank_processor = LowRankFactorization(epsilon=epsilon, device=device)
    compressed_model, stats = low_rank_processor.compress_model(base_model, epsilon)

    # 将统计信息添加到模型作为属性
    compressed_model.compression_stats = stats

    return compressed_model