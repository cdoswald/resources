"""PyTorch-based decoder model classes."""

from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_device, get_np_torch_dtype_map

DEVICE = get_device()


class MLPDecoder(nn.Module):
    """Fully-connected multi-layer perception (MLP) decoder."""

    def __init__(
        self,
        n_input_channels: int,
        n_final_output_channels: int,
        hidden_channels: List[int] = [64, 32, 16],
        activ_func: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """Initialize and construct network architecture."""
        super().__init__()
        layers = []
        for n_output_channels in hidden_channels:
            fc_layer = nn.Linear(n_input_channels, n_output_channels)
            layers += [fc_layer, activ_func()]
            n_input_channels = n_output_channels
        layers += [nn.Linear(n_input_channels, n_final_output_channels)]
        self.decoder = nn.Sequential(*layers).to(DEVICE)

    def forward(self, encoded_obs: torch.tensor):
        return self.decoder(encoded_obs)


class UNetDecoderBlock(nn.Module):
    """Upsample using 2x2 transposed convolution layers, concatenate with skip
    connections, and then apply 2 successive layers of 3x3 convolutions and ReLU."""

    def __init__(self, n_input_channels: int, n_output_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            n_input_channels,
            n_input_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.decode = nn.Sequential(
            nn.Conv2d(n_input_channels, n_output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_output_channels, n_output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.upsample(x)
        # Crop skip tensor (based on
        # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py)
        height_diff = skip.size()[2] - x.size()[2]
        width_diff = skip.size()[3] - x.size()[3]
        pad = [
            width_diff // 2, width_diff - width_diff // 2, # Pad last dimension
            height_diff // 2, height_diff - height_diff // 2, # Pad second-to-last dimension
        ]
        x = F.pad(x, pad)
        x = torch.cat([x, skip], dim=1)
        return self.decode(x)


class UNetDecoder(nn.Module):
    """U-Net style decoder model."""

    def __init__(self, n_input_channels: int, n_final_output_channels: int):
        super().__init__()
        self.decode_1 = UNetDecoderBlock(n_input_channels, 512)
        self.decode_2 = UNetDecoderBlock(512, 256)
        self.decode_3 = UNetDecoderBlock(256, 128)
        self.decode_4 = UNetDecoderBlock(128, 64)
        self.decode_last = nn.Conv2d(64, n_final_output_channels, kernel_size=1) #1x1 Conv2d

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]):
        assert len(skips) == 4
        x = self.decode_1(x, skips[-1])
        x = self.decode_2(x, skips[-2])
        x = self.decode_3(x, skips[-3])
        x = self.decode_4(x, skips[-4])
        return self.decode_last(x)
