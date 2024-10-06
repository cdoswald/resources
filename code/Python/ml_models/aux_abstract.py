"""Abstract base class template for self-supervised auxiliary models."""

from abc import ABC, abstractmethod
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

from ..utils import get_device, get_np_torch_dtype_map

DEVICE = get_device()


class AuxModelAbstract(nn.Module, ABC):
    """Abstract base class for all auxiliary models."""

    @abstractmethod
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.loss_func = None

    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create modified observations and new labels for SSL task. Then
        compute forward pass through encoder and decoder. Returns a tuple of
        (model_predictions, true_labels, modified_images)."""
        return preds, labels, SSL_obs

    @abstractmethod
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations using model-specific encoder."""
        return encoded_obs

    def compute_encoded_vec_norm(self, obs: torch.Tensor, norm: int = 2):
        """Compute norm of encoded vector."""
        return torch.norm(self.encode(obs).detach().flatten(start_dim=1), p=norm, dim=-1)
    
    def freeze_decoder(self, obs: torch.Tensor) -> None:
        """Freeze all weights in decoder to prepare for test-time updates to encoder only."""
        for param in self.decoder.parameters():
            param.requires_grad = False

    def _shuffle(self, obs, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly shuffle observations and labels."""
        idxs = torch.randperm(obs.shape[0])
        return obs[idxs], labels[idxs]

    def train_model(
        self,
        train_dl: DataLoader,
        eval_dl: DataLoader,
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
        convert_labels_to_long: bool = False,
    ) -> None:
        """Train self-supervised model."""
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
                train_norm_loss = torch.sum(self.compute_encoded_vec_norm(X))
                if convert_labels_to_long:
                    train_labels = train_labels.long()
                train_loss = self.loss_func(train_preds, train_labels) + train_norm_loss
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
                    eval_norm_loss = torch.sum(self.compute_encoded_vec_norm(X))
                    if convert_labels_to_long:
                        eval_labels = eval_labels.long()
                    eval_loss = self.loss_func(eval_preds, eval_labels) + eval_norm_loss
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

    def step_encoder(
        self,
        obs: torch.Tensor,
        n_steps: int,
        lr: float,
        max_grad_norm: Optional[float] = 1.0,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        verbose: bool = False,
        convert_labels_to_long: bool = False,
        return_losses: bool = False,
    ) -> Optional[List[float]]:
        """Update encoder for n gradient steps."""
        losses = []
        optimizer = optimizer(self.encoder.parameters(), lr=lr) # NOTE: encoder only
        self.encoder.train()
        self.decoder.eval()
        for step_idx in range(n_steps):
            SSL_preds, SSL_labels, _ = self.forward(obs)
            if convert_labels_to_long:
                SSL_labels = SSL_labels.long()
            loss = self.loss_func(SSL_preds, SSL_labels)
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(self.encoder.parameters(), max_norm=max_grad_norm) #NOTE: encoder only
            optimizer.step()
            losses.append(loss.item())
            if verbose:
                print(f"Step {step_idx+1} loss: {loss.item()}")
        if return_losses:
            return losses
