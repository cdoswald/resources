"""Functions for MuJoCo environments."""

from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from .wrappers import ConvertToTensorWrapper


def setup_env(
    task_name: str,
    run_name: str,
    render_mode="rgb_array",
    record_video: bool = False,
    videos_dir: Optional[str] = None,
):
    """Set up MuJoCo environment with wrappers."""
    env = gym.make(task_name, render_mode=render_mode)
    env = ConvertToTensorWrapper(env)
    if record_video:
        if videos_dir is None:
            raise ValueError('Arg "videos_dir" cannot be None if "record_video" is True.')
        env = RecordVideo(
            env,
            video_folder=videos_dir,
            name_prefix=run_name,
            episode_trigger=lambda x: True,
        )
    print(f"Observation space sample shape: {env.observation_space.sample().shape}")
    return env
