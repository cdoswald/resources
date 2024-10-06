"""Custom wrapper classes for Gym-based environments.

Note that all environments should have shape (n_envs, obs_space_dims) and
(n_envs, action_space_dims).
"""

from typing import List

import numpy as np
import torch

import gymnasium as gym


class ManiskillImageProcessorWrapper(gym.ObservationWrapper):
    """Image processor wrapper for Maniskill environments.
    
    Modifies observation space to handle concatenated images from each
    combination of camera view and camera mode specified by user.

    Args
        env: ManiSkill environment
        camera_views: list of camera views to use; valid options for ManiSkill
            are {"base_camera"}
        camera_modes: list of camera modes to use; valid options for ManiSkill
            are {"rgb", "depth", "segmentation"}

    Returns
        observation: images concatenated along channel dimension
    """

    def __init__(self, env, camera_views: List[str], camera_modes: List[str]):
        super().__init__(env)
        self.camera_views = camera_views
        self.camera_modes = camera_modes

        # Define new observation space (concatenated on channel dimension)
        sample_image = self._process_images(self.env.reset()[0]) # Reset() returns 2-tuple of dicts with dict keys (['agent', 'extra', 'sensor_param', 'sensor_data'], ['reconfigure'])
        print(f'Sample image shape: {sample_image.shape}')
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=sample_image.shape,
            dtype=np.uint8,
        )

    def _process_images(self, obs):
        """Extract and concatenate images from each combination of camera view
        and mode."""
        images = []
        for camera_view in self.camera_views:
            for camera_mode in self.camera_modes:
                images.append(obs['sensor_data'][camera_view][camera_mode])
        return torch.cat(images, dim=-1).permute(0, 3, 1, 2) # BxCxHxW to match PyTorch format

    def observation(self, obs):
        return self._process_images(obs)


class ConvertToTensorWrapper(gym.Wrapper):
    """Wrapper to convert observation from numpy array to torch tensor.
    Observation space has shape (n_envs, *obs_space_dims) and action space
    has shape (n_envs, *action_space_dims)."""

    def __init__(self, env):
        super().__init__(env)
        # Observation space
        if len(self.observation_space.shape) == 1:
            self.obs_space_dims = self.observation_space.shape[0]
            new_shape = (1, self.obs_space_dims) # Set n_envs dim to 1
            self.observation_space = gym.spaces.Box(
                low=self.observation_space.low.reshape(new_shape),
                high=self.observation_space.high.reshape(new_shape),
                shape=new_shape,
                dtype=self.observation_space.dtype,
            )
        else:
            self.obs_space_dims = self.observation_space.shape[1:] # First dim is n_envs
            # Keep self.observation_space the same

        # Action space
        if len(self.action_space.shape) == 1:
            self.action_space_dims = self.action_space.shape[0]
            new_shape = (1, self.action_space_dims)
            self.action_space = gym.spaces.Box(
                low=self.action_space.low.reshape(new_shape),
                high=self.action_space.high.reshape(new_shape),
                shape=new_shape,
                dtype=self.action_space.dtype,
            )
        else:
            self.action_space_dims = self.action_space.shape[1:]
            # Keep self.action_space the same

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return obs.float().clone().detach(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward)
        if not isinstance(terminated, torch.Tensor):
            terminated = torch.tensor(terminated)
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.tensor(truncated)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return obs.float().clone().detach(), reward, terminated, truncated, info

    def sample_obs_tensor(self):
        """Alternative to env.observation_space.sample() that returns a torch
        tensor instead of a numpy array."""
        # Required to avoid dropping n_env dimension at index 0 when n_env == 1
        obs_space_shape = self.observation_space.shape
        sample_obs = self.observation_space.sample().reshape(obs_space_shape)
        return torch.tensor(sample_obs, dtype=torch.float32).detach()

    def sample_action_tensor(self):
        """Alternative to env.action_space.sample() that returns a torch
        tensor instead of a numpy array."""
        # Required to avoid dropping n_env dimension at index 0 when n_env == 1
        action_space_shape = self.action_space.shape
        sample_obs = self.action_space.sample().reshape(action_space_shape)
        return torch.tensor(sample_obs, dtype=torch.float32).detach()


class CPUEnvWrapper(gym.Wrapper):
    """Wrapper to convert ManiSkill GPU-based environment to CPU-based environment
    for use with Stable Baselines3.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
            # reward = reward.cpu().numpy()
            # terminated = terminated.cpu().numpy()
            # truncated = truncated.cpu().numpy()
        return obs, reward, terminated, truncated, info
