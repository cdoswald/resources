"""Soft actor-critic (SAC) classes."""

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from ..utils import get_device

DEVICE = get_device()


class Actor(nn.Module):
    """Actor network for soft actor-critic (SAC) agent.
    
    Based on CleanRL implementation 
    (https://github.com/vwxyzjn/cleanrl).
    """

    def __init__(
        self,
        obs_encoder_channels: int,
        action_space_channels: int,
        action_space_high: torch.Tensor,
        action_space_low: torch.Tensor,
        hidden_channels: List[int] = [256, 256, 256],
        activ_function: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        layers = []
        n_input_channels = obs_encoder_channels
        for n_output_channels in hidden_channels:
            fc_layer = nn.Linear(n_input_channels, n_output_channels)
            layers += [fc_layer, activ_function()]
            n_input_channels = n_output_channels
        layers_mean = layers + [nn.Linear(n_input_channels, action_space_channels)]
        layers_log_std = layers + [nn.Linear(n_input_channels, action_space_channels)]
        self.mean_decoder = nn.Sequential(*layers_mean)
        self.log_std_decoder = nn.Sequential(*layers_log_std)
        # action rescaling
        action_scale = torch.tensor((action_space_high - action_space_low)/ 2.0).float()[0] # All envs have same action spaces
        action_bias = torch.tensor((action_space_high + action_space_low) / 2.0).float()[0] # All envs have same action spaces
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(
        self,
        encoded_obs: torch.Tensor,
        log_std_min_max: Optional[List[int]] = [-5, 2],
    ) -> Tuple[float, float]:
        mean = self.mean_decoder(encoded_obs)
        log_std = self.log_std_decoder(encoded_obs)
        if log_std_min_max:
            log_std = torch.tanh(log_std)
            log_std = (
                log_std_min_max[0] +
                0.5 * (log_std_min_max[1] - log_std_min_max[0]) * (log_std + 1)
            )
        return mean, log_std

    def get_action(
        self,
        encoded_obs: torch.Tensor,
        log_std_min_max: Optional[List[int]] = [-5, 2],
    ):
        mean, log_std = self.forward(encoded_obs, log_std_min_max)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # Sample using reparameterization trick
        # eps = torch.randn_like(mean).to(DEVICE)
        # print(f"Devices: mean: {mean.device}; std: {std.device}; eps: {eps.device}")
        
        # x_t = mean + std * eps # Implement reparameterization trick manually
        # to reduce computation time associated with initializing new torch distribution
        y_t = torch.tanh(x_t) # Scale sampled action to [-1, 1]
        # print(
        #     f'Checkpoint: y_t shape = {y_t.shape} '
        #     f'\naction_scale shape = {self.action_scale.shape} '
        #     f'\naction_bias shape = {self.action_bias.shape}'
        # )
        action = y_t * self.action_scale + self.action_bias # Rescale to original action space
        ## TODO: line 81 throwing error when env > 1 and train_policy_simple.py
        log_prob = normal.log_prob(x_t)
        # Compute log likelihood manually (since implementing reparam trick manually)
        # log_prob = (-0.5) * (torch.log(torch.tensor(2*torch.pi)) + 2*torch.log(std) + ((x_t - mean) / std)**2)
        # Adjust log_prob to account for tanh scaling (uses Jacobian for COV)
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    


        # """Option 1"""
        # eps = torch.randn_like(mean).to(DEVICE)
        # x_t = mean + std * eps # Implement reparameterization trick manually
        # log_prob = (-0.5) * (torch.log(torch.tensor(2*torch.pi)) + 2*torch.log(std) + ((x_t - mean) / std)**2)

        # """Option 2"""
        # normal = torch.distributions.Normal(mean, std)
        # x_t = normal.rsample() # Sample using reparameterization trick
        # log_prob = normal.log_prob(x_t)
