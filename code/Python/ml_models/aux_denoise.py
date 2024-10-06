"""Self-supervised auxiliary model for image denoising."""

import json
import os
import time
from typing import List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from .encoders import UNetEncoder
from .decoders import UNetDecoder
from ..utils import get_device, get_np_torch_dtype_map

DEVICE = get_device()


class AuxModelImageDenoising(nn.Module):
    """Self-supervised auxiliary model for image denoising task."""

    def __init__(
        self,
        n_input_channels: int,
        sample_input: torch.Tensor,
        noise_stddevs: List[float] = [10.0, 20.0],
    ) -> None:
        """Initialize encoder and decoder networks."""
        super().__init__()
        self.noise_stddevs = noise_stddevs
        self.encoder = UNetEncoder(n_input_channels).to(DEVICE)
        # Run random sample through encoder to get input dims for decoder
        n_encoder_channels = self.encoder(sample_input, return_skips=False).shape[1] # Not flattened
        n_output_channels = n_input_channels # Same as input channels for denoising
        self.decoder = UNetDecoder(n_encoder_channels, n_output_channels).to(DEVICE)

    def _add_noise(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add random noise to observations. Returns a tuple of
        (noisy_images, true_images), where noisy and true images have
        shape (batch_size*len(noise_stddevs), channels, height, width)."""
        original_shape = obs.shape
        batch_size = original_shape[0]
        channels = original_shape[1]
        new_batch_size = batch_size * len(self.noise_stddevs)
        noisy_images_all = torch.empty(
            (new_batch_size, *original_shape[1:]), dtype=torch.float32
        )
        true_images_all = torch.empty_like(noisy_images_all, dtype=torch.float32)
        for idx, noise_stddev in enumerate(self.noise_stddevs):
            for channel in range(channels):
                obs_channel = obs[:, channel, :, :]
                # Get min and max values for clipping
                obs_min = torch.min(obs_channel)
                obs_max = torch.max(obs_channel)
                # Convert original uint8 images to float
                obs_float = obs_channel.float()
                # Generate and add noise to channel
                normal_dist = torch.distributions.Normal(0., scale=noise_stddev)
                noise = normal_dist.sample(obs_float.shape).to(DEVICE)
                noisy_images_channel = torch.clip(
                    obs_float + noise, obs_min, obs_max
                )
                # Save images to arrays
                start_idx = idx*batch_size
                end_idx = (idx+1)*batch_size
                noisy_images_all[start_idx:end_idx, channel, ...] = noisy_images_channel
                true_images_all[start_idx:end_idx, channel, ...] = obs_channel
        return noisy_images_all.to(DEVICE), true_images_all.to(DEVICE)

    def _shuffle(self, obs, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly shuffle observations and labels."""
        idxs = torch.randperm(obs.shape[0])
        return obs[idxs], labels[idxs]

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

    def train_model(
        self,
        train_dl: DataLoader,
        eval_dl: DataLoader,
        loss_func: torch.nn.functional,
        max_epochs: int,
        batch_size: int,
        lr: float,
        save_model_dir: str,
        save_model_suffix: str = "",
        metadata_dir: Optional[str] = None,
        metadata_suffix: Optional[str] = None,
        max_grad_norm: Optional[float] = 1.0,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        verbose: bool = True,
    ) -> None:
        """Train and evaluate self-supervised model."""
        optimizer = optimizer(self.parameters(), lr=lr)
        train_loss_dict = {epoch:[] for epoch in range(max_epochs)}
        eval_loss_dict = {epoch:[] for epoch in range(max_epochs)}
        best_eval_loss = None
        for epoch in range(max_epochs):
            epoch_start = time.time()
            # Training loop
            self.encoder.train()
            self.decoder.train()
            for idx, (X, y) in enumerate(train_dl):
                train_preds, train_labels, _ = self.forward(X)
                train_loss = loss_func(train_preds, train_labels)
                train_loss_dict[epoch].append(train_loss.item())
                optimizer.zero_grad()
                train_loss.backward()
                if max_grad_norm is not None:
                    clip_grad_norm_(self.parameters(), max_norm=max_grad_norm)
                optimizer.step()
            # Evaluation loop
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                for idx, (X, y) in enumerate(eval_dl):
                    eval_preds, eval_labels, _ = self.forward(X)
                    eval_loss = loss_func(eval_preds, eval_labels)
                    eval_loss_dict[epoch].append(eval_loss.item())
            # Save mtrain_modelodel if improving
            eval_epoch_loss = np.mean(eval_loss_dict[epoch])
            if best_eval_loss is None or eval_epoch_loss < best_eval_loss:
                best_eval_loss = eval_epoch_loss
                outpath = os.path.join(save_model_dir, f"aux_model{save_model_suffix}.pt")
                torch.save(self.state_dict(), outpath)
                print(f"Saved model at epoch {epoch+1}.")
            # Save loss data
            if metadata_dir is None:
                metadata_dir = save_model_dir
            if metadata_suffix is None:
                metadata_suffix = save_model_suffix
            with open(os.path.join(metadata_dir, f"aux_train_loss{metadata_suffix}.json"), "w") as file:
                json.dump(train_loss_dict, file, indent=4)
            with open(os.path.join(metadata_dir, f"aux_eval_loss{metadata_suffix}.json"), "w") as file:
                json.dump(eval_loss_dict, file, indent=4)
            # Print progress
            if verbose:
                epoch_end = time.time()
                print(
                    f"Epoch {epoch+1} eval loss: {round(eval_epoch_loss, 3)} "
                    f"({round(epoch_end - epoch_start, 2)} seconds)"
                )
