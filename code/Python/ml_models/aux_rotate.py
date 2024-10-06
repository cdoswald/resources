"""Self-supervised auxiliary model for image rotation."""

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

from .encoders import CNNEncoder
from .decoders import MLPDecoder
from ..utils import get_device, get_np_torch_dtype_map

DEVICE = get_device()


class AuxModelImageRotation(nn.Module):
    """Self-supervised auxiliary model for image rotation task."""

    def __init__(
        self,
        n_input_channels: int,
        sample_input: torch.Tensor,
        angles: List[int] = [0, 90, 180, 270],
    ) -> None:
        """Initialize encoder and decoder networks."""
        super().__init__()
        self.angles = angles
        self.encoder = CNNEncoder(n_input_channels).to(DEVICE)
        # Run random sample through encoder to get input dims for decoder
        n_encoder_channels = self.encoder(sample_input).shape[-1] # Flattened (axis should be -1)
        n_output_channels = len(angles) # multi-classification
        self.decoder = MLPDecoder(n_encoder_channels, n_output_channels).to(DEVICE)

    def _rotate_images(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate images by specified degrees. Returns a tuple of
        (rotated_images, angle_labels), where rotated images have shape
        (batch_size*len(angles), channels, height, width).
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
        # Create labels array
        labels = torch.empty((batch_size * len(self.angles)), ).to(DEVICE)
        for idx, angle in enumerate(temp.keys()):
            labels[idx*batch_size : (idx+1)*batch_size] = int(angle / 90) #TODO: must use sparse categorical cross-entropy loss
        # Concatenate image arrays
        all_rotated_obs = torch.cat(list(temp.values()), axis=0)
        return all_rotated_obs, labels

    def _shuffle(self, obs, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly shuffle observations and labels."""
        idxs = torch.randperm(obs.shape[0])
        return obs[idxs], labels[idxs]

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
                train_loss = loss_func(train_preds, train_labels.long()) # NOTE: labels converted to long for classification
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
                    eval_loss = loss_func(eval_preds, eval_labels.long()) # NOTE: labels converted to long for classification
                    eval_loss_dict[epoch].append(eval_loss.item())
            # Save model if improving
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
