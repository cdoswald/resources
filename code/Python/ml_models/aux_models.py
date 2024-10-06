"""Self-supervised auxiliary models.

Classes:
    AuxModelImageRotation
    AuxModelImageDenoising
    AuxModelImageColorization
"""

import json
import os
import time
from typing import List, Optional, Tuple, Type

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from .aux_abstract import AuxModelAbstract
from .decoders import MLPDecoder, UNetDecoder
from .encoders import CNNEncoder, UNetEncoder
from ..utils import get_device, get_np_torch_dtype_map

DEVICE = get_device()


class AuxModelImageRotation(AuxModelAbstract):
    """Self-supervised auxiliary model for image rotation task."""

    def __init__(
        self,
        sample_input: torch.Tensor,
        angles: List[int] = [0, 90, 180, 270],
        loss_func: torch.nn.functional = F.cross_entropy,
    ) -> None:
        """Initialize encoder and decoder networks."""
        super().__init__()
        self.n_input_channels = sample_input.shape[1]
        self.angles = angles
        self.loss_func = loss_func
        self.encoder = CNNEncoder(self.n_input_channels).to(DEVICE)
        # Run random sample through encoder to get input dims for decoder
        n_encoder_channels = self.encoder(sample_input).shape[-1] # Flattened (axis should be -1)
        n_output_channels = len(angles) # multi-classification
        self.decoder = MLPDecoder(n_encoder_channels, n_output_channels).to(DEVICE)

    def _rotate_images(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate images by specified degrees. Returns a tuple of
        (rotated_images, angle_labels), where rotated images have shape
        (batch_size, channels, height, width).
        """
        batch_size = obs.shape[0]
        n_channels = obs.shape[1]
        height = obs.shape[2]
        width = obs.shape[3]
        center = (height // 2, width // 2)
        temp = {}
        for angle in self.angles:
            rotated_obs = torch.empty_like(obs)
            for idx in range(batch_size):
                for channel in range(n_channels):
                    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rot_image = cv2.warpAffine(
                        obs[idx, channel, :, :].detach().cpu().numpy(),
                        rot_matrix,
                        (height, width),
                        flags=cv2.INTER_LINEAR,
                    )
                    rotated_obs[idx, channel, :, :] = torch.from_numpy(rot_image).to(DEVICE)
            temp[angle] = rotated_obs
        # Select one augmented image for each original image (random angle)
        labels = torch.randint(len(self.angles), size=(batch_size, )).to(DEVICE)
        rotated_obs = torch.zeros_like(obs)
        for idx in range(batch_size):
            selected_angle = (labels[idx] * 90).item()
            rotated_obs[idx] = temp[selected_angle][idx]
        return rotated_obs, labels

    def forward(
        self,
        obs: torch.Tensor,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create modified observations and new labels for SSL task. Then
        compute forward pass through encoder and decoder. Returns a tuple of
        (model_predictions, true_labels, modified_images)."""
        if shuffle:
            SSL_obs, labels = self._shuffle(*self._rotate_images(obs))
        else:
            SSL_obs, labels = self._rotate_images(obs)
        return (self.decoder(self.encoder(SSL_obs)), labels, SSL_obs)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations using model-specific encoder."""
        SSL_obs = self._rotate_images(obs)[0]
        return self.encoder(SSL_obs)


class AuxModelImageDenoising(AuxModelAbstract):
    """Self-supervised auxiliary model for image denoising task."""

    def __init__(
        self,
        sample_input: torch.Tensor,
        noise_stddevs: List[float] = [10.0, 20.0, 30.0, 40.0, 50.0],
        loss_func: torch.nn.functional = F.mse_loss,
    ) -> None:
        """Initialize encoder and decoder networks."""
        super().__init__()
        self.n_input_channels = sample_input.shape[1]
        self.noise_stddevs = noise_stddevs
        self.loss_func = loss_func
        self.encoder = UNetEncoder(self.n_input_channels).to(DEVICE)
        # Run random sample through encoder to get input dims for decoder
        n_encoder_channels = self.encoder(sample_input, return_skips=False).shape[1] # Not flattened
        n_output_channels = self.n_input_channels # Same as input channels for denoising
        self.decoder = UNetDecoder(n_encoder_channels, n_output_channels).to(DEVICE)

    def _add_noise(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add random noise to observations. Returns a tuple of
        (noisy_images, true_images), where noisy and true images have
        shape (batch_size, channels, height, width)."""
        batch_size = obs.shape[0]
        channels = obs.shape[1]
        height, width = obs.shape[2], obs.shape[3]
        noisy_images = torch.empty_like(obs, dtype=torch.float32)
        true_images = obs.clone()
        for idx in range(batch_size):
            for channel in range(channels):
                obs_channel = obs[idx, channel, :, :]
                # Get min and max values for clipping
                obs_min = torch.min(obs_channel)
                obs_max = torch.max(obs_channel)
                # Convert original uint8 images to float
                obs_float = obs_channel.float()
                # Generate and add noise to channel
                selected_stddev = np.random.choice(self.noise_stddevs)
                normal_dist = torch.distributions.Normal(0., scale=selected_stddev)
                noise = normal_dist.sample(obs_float.shape).to(DEVICE)
                noisy_images_channel = torch.clip(
                    obs_float + noise, obs_min, obs_max
                )
                # Save images to arrays
                noisy_images[idx, channel, ...] = noisy_images_channel
        return noisy_images.to(DEVICE), true_images.to(DEVICE)

    def forward(
        self,
        obs: torch.Tensor,
        rescale: bool = True,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create modified observations and new labels for SSL task. Then
        compute forward pass through encoder and decoder. Returns a tuple of
        (model_predictions, true_labels, modified_images)."""
        if shuffle:
            SSL_obs, labels = self._shuffle(*self._add_noise(obs))
        else:
            SSL_obs, labels = self._add_noise(obs)
        temp = self.encoder(SSL_obs, return_skips=True)
        preds, labels = (
            self.decoder(*self.encoder(SSL_obs, return_skips=True)), labels
        )
        if rescale:
            labels_min, labels_max = torch.min(labels), torch.max(labels)
            labels_range = labels_max - labels_min
            preds_min, preds_max = torch.min(preds), torch.max(preds)
            preds_range = preds_max - preds_min
            preds = (
                (preds - preds_min) / preds_range # Normalize to [0, 1]
                * labels_range + labels_min # Scale to original image
            )
        return preds, labels, SSL_obs

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations using model-specific encoder."""
        SSL_obs = self._add_noise(obs)[0]
        return self.encoder(SSL_obs)


class AuxModelImageColorization(AuxModelAbstract):
    """Self-supervised auxiliary model for image colorization task."""

    def __init__(
        self,
        sample_input: torch.Tensor,
        loss_func: torch.nn.functional = F.mse_loss,
    ) -> None:
        """Initialize encoder and decoder networks."""
        super().__init__()
        self.n_color_channels = sample_input.shape[1]
        self.loss_func = loss_func
        self.encoder = UNetEncoder(self.n_color_channels).to(DEVICE)
        # Run random sample through encoder to get input dims for decoder
        gray_sample, _ = self._convert_to_gray(sample_input)
        n_encoder_channels = self.encoder(gray_sample, return_skips=False).shape[1] # Not flattened
        self.decoder = UNetDecoder(n_encoder_channels, self.n_color_channels).to(DEVICE)

    def _convert_to_gray(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert observations from RGB to grayscale. Returns a tuple of
        (gray_images, true_images), where gray and true images have
        shape (batch_size, channels, height, width)."""
        rgb_weights = torch.tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1).to(DEVICE) # weighting based on OpenCV docs ("cvtColor")
        gray_images = torch.sum(obs * rgb_weights, axis=1).unsqueeze(1).repeat(
            1, self.n_color_channels, 1, 1 # Duplicate grayscale values across n color channels
        )
        return gray_images, obs

    def forward(
        self,
        obs: torch.Tensor,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create modified observations and new labels for SSL task. Then
        compute forward pass through encoder and decoder. Returns a tuple of
        (model_predictions, true_labels, modified_images)."""
        if shuffle:
            SSL_obs, labels = self._shuffle(*self._convert_to_gray(obs))
        else:
            SSL_obs, labels = self._convert_to_gray(obs)
        preds = self.decoder(*self.encoder(SSL_obs, return_skips=True))
        return preds, labels, SSL_obs

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations using model-specific encoder."""
        SSL_obs = self._convert_to_gray(obs)[0]
        return self.encoder(SSL_obs)
