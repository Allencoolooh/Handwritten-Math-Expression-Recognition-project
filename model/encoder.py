# model/encoder.py
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
import torchvision.models as models


class ConvBlock(nn.Module):
    """
    一个小的卷积块：Conv2d + BatchNorm2d + ReLU + 可选的 MaxPool2d
    用来堆积成 CNN 编码器。
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | Tuple[int, int] = 3,
            stride: int | Tuple[int, int] = 1,
            padding: int | Tuple[int, int] = 1,
            pool: Tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool) if pool is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        if self.pool is not None:
            x = self.pool(x)
        return x


class CNNEncoder(nn.Module):
    """
    将输入图像 (B, 1, H, W) 编码成序列特征 (B, T, D)，供 Transformer Encoder 使用。

    设计思路：
    - 一组 2D 卷积 + 池化，把 H, W 缩小、通道数增加；
    - 最后把高度维 H 压到通道里，把宽度维 W 当作“时间步 T”；
    - 用一个线性层把通道维投影到 Config.D_MODEL，作为最终特征维度。

    默认输入尺寸：H = 64, W = 512（在 dataset 里已经统一了）
    """

    def __init__(
            self,
            in_channels: int = Config.NUM_CHANNELS,  # 通常是 1（灰度图）
            d_model: int = Config.D_MODEL,
    ) -> None:
        super().__init__()

        # 一共四个卷积块，逐步压缩 H / W，增加通道数
        # 输入: (B, 1, 64, 512)

        # block1: (B, 1, 64, 512) -> (B, 64, 32, 256)
        self.block1 = ConvBlock(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            pool=(2, 2),  # 高度/宽度都缩小一半
        )

        # block2: (B, 64, 32, 256) -> (B, 128, 16, 128)
        self.block2 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            pool=(2, 2),
        )

        # block3: (B, 128, 16, 128) -> (B, 256, 8, 128)
        # 只在高度方向做池化（2,1），宽度不再缩小，保留更多时间步
        self.block3 = ConvBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            pool=(2, 1),
        )

        # block4: (B, 256, 8, 128) -> (B, 256, 4, 128)
        self.block4 = ConvBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            pool=(2, 1),
        )

        # 现在特征图尺寸大约是 (B, 256, H=4, W=128)
        # 我们把 H=4 压到通道里: C' = 256 * 4 = 1024
        self._height_after_cnn = 4
        self._channels_after_cnn = 256
        cnn_out_dim = self._channels_after_cnn * self._height_after_cnn  # 1024

        # 线性投影到 d_model，作用在每个时间步上
        self.proj = nn.Linear(cnn_out_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: (B, C_in, H, W)，例如 (B, 1, 64, 512)
        输出：
            seq: (B, T, D)，T 对应卷积之后的宽度（例如 128），D = d_model
        """
        B, C, H, W = x.shape
        # 1. CNN 下采样
        x = self.block1(x)  # -> (B, 64, 32, 256)
        x = self.block2(x)  # -> (B, 128,16,128)
        x = self.block3(x)  # -> (B, 256, 8, 128)
        x = self.block4(x)  # -> (B, 256, 4, 128)

        B, C, H, W = x.shape  # 预期 H=4, C=256, W≈128

        # 2. 把高度维 H 合并到通道维： (B, C, H, W) -> (B, C*H, W)
        x = x.view(B, C * H, W)  # (B, 256*4, 128) = (B, 1024, 128)

        # 3. 把宽度 W 当作时间步 T，调换维度 -> (B, T, C*H)
        x = x.permute(0, 2, 1)  # (B, T=128, 1024)

        # 4. 线性映射到 d_model 维度 -> (B, T, D)
        seq = self.proj(x)  # (B, T, d_model)

        return seq


class ResNetEncoder(nn.Module):
    """
    用 ResNet-18 做图像特征提取的编码器：
    输入:  (B, C_in, H, W)
    输出:  (B, T, D)  其中 T≈W/32, D=Config.D_MODEL
    """

    def __init__(
            self,
            in_channels: int = Config.NUM_CHANNELS,
            d_model: int = Config.D_MODEL,
    ) -> None:
        super().__init__()

        # 1. 建一个 resnet18，去掉 avgpool 和 fc
        base = models.resnet18(weights=None)
        # 替换首层 conv，支持 1 通道输入
        base.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # 保留到 layer4 之前 (conv1~layer4)
        self.backbone = nn.Sequential(*list(base.children())[:-2])  # 去掉 avgpool, fc

        self.d_model = d_model

        # 2. 用一个 dummy 输入推一遍，确定 C, H'
        #    假定你的图片在 dataset 里已经统一为 (H, W) = (Config.IMG_HEIGHT, Config.MAX_WIDTH)
        dummy = torch.zeros(1, in_channels, Config.IMG_HEIGHT, Config.MAX_WIDTH)
        with torch.no_grad():
            feat = self.backbone(dummy)   # (1, C, H', W')
            _, C, H, W = feat.shape

        in_dim = C * H   # 我们把高度 H' 合到通道里，所以线性层输入为 C*H'

        # 3. 现在在 __init__ 里就把 proj 建好，这样 encoder.proj 始终存在
        self.proj = nn.Linear(in_dim, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        return: (B, T, D)
        """
        B = x.size(0)

        # 1. ResNet 提特征
        feat = self.backbone(x)      # (B, C, H', W')
        B, C, H, W = feat.shape

        # 2. 合并高度维: (B, C, H, W) -> (B, C*H, W)
        feat = feat.view(B, C * H, W)

        # 3. 把 W 当作时间步 T: (B, C*H, W) -> (B, T=W, C*H)
        feat = feat.permute(0, 2, 1)  # (B, T, C*H)

        # 4. 线性映射到 d_model
        seq = self.proj(feat)        # (B, T, d_model)

        return seq
