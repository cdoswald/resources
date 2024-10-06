"""PyTorch-based encoder model classes."""

from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_device, get_np_torch_dtype_map

DEVICE = get_device()

class MLPEncoder(nn.Module):
    """MLP encoder for state information."""

    def __init__(
        self,
        n_input_channels: int,
        hidden_channels: List[int] = [256, 256, 256],
        activ_func: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        layers = []
        for n_output_channels in hidden_channels:
            linear_layer = nn.Linear(n_input_channels, n_output_channels)
            layers += [linear_layer, activ_func()]
            n_input_channels = n_output_channels
        self.linear = nn.Sequential(*layers).to(DEVICE)

    def forward(self, obs: torch.Tensor):
        return self.linear(obs)


class CNNEncoder(nn.Module):
    """CNN encoder for visual state information (e.g., RGBD)."""

    def __init__(
        self,
        n_input_channels: int,
        hidden_channels: List[int] = [64, 64, 128, 256, 512, 1024],
        conv2d_kernel: int = 3,
        conv2d_padding: int = 1,
        maxpool2d_kernel: int = 2,
        maxpool2d_stride: int = 2,
        activ_func: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """Initialize and construct network architecture."""
        super().__init__()

        # Create network architecture
        layers = []
        for n_output_channels in hidden_channels:
            conv_layer = nn.Conv2d(
                n_input_channels,
                n_output_channels,
                kernel_size=conv2d_kernel,
                padding=conv2d_padding,
            )
            max_pool_layer = nn.MaxPool2d(
                kernel_size=maxpool2d_kernel,
                stride=maxpool2d_stride,
            )
            layers += [conv_layer, max_pool_layer, activ_func()]
            n_input_channels = n_output_channels
        self.cnn = nn.Sequential(*layers).to(DEVICE)

    def forward(self, obs: torch.tensor, flatten: bool = True) -> torch.Tensor:
        if not flatten:
            return self.cnn(obs)
        return self.cnn(obs).flatten(start_dim=1) # Preserve batch dimension


class UNetEncoderBlock(nn.Module):
    """Apply 2 successive layers of 3x3 convolutions and ReLU activation."""

    def __init__(self, n_input_channels: int, n_output_channels: int):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(n_input_channels, n_output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_output_channels, n_output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.tensor, maxpool_first=False):
        if maxpool_first:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return self.encode(x)


class UNetEncoder(nn.Module):
    """U-Net style encoder model."""

    def __init__(self, n_input_channels: int):
        super().__init__()
        self.encode_1 = UNetEncoderBlock(n_input_channels, 64)
        self.encode_2 = UNetEncoderBlock(64, 128)
        self.encode_3 = UNetEncoderBlock(128, 256)
        self.encode_4 = UNetEncoderBlock(256, 512)
        self.encode_5 = UNetEncoderBlock(512, 1024)
        self.encode_6 = UNetEncoderBlock(1024, 2048)
        self.encode_last = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2048, 4096, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4096, 4096, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, obs: torch.tensor, return_skips: bool = False):
        if return_skips:
            x1 = self.encode_1(obs)
            x2 = self.encode_2(x1, maxpool_first=True)
            x3 = self.encode_3(x2, maxpool_first=True)
            x4 = self.encode_4(x3, maxpool_first=True)
            x5 = self.encode_5(x4, maxpool_first=True)
            x6 = self.encode_6(x5, maxpool_first=True)
            return self.encode_last(x6), [x1, x2, x3, x4, x5, x6]
        else:
            x = self.encode_1(obs)
            x = self.encode_2(x, maxpool_first=True)
            x = self.encode_3(x, maxpool_first=True)
            x = self.encode_4(x, maxpool_first=True)
            x = self.encode_5(x, maxpool_first=True)
            x = self.encode_6(x, maxpool_first=True)
            return self.encode_last(x)
