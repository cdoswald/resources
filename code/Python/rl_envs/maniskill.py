"""Functions for ManiSkill environments."""

from typing import Optional

import gymnasium as gym

import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode

from .wrappers import ConvertToTensorWrapper, ManiskillImageProcessorWrapper


def setup_env(
    task_name: str,
    run_name: str,
    num_envs: int = 4,
    obs_mode: str = "rgbd",
    render_mode: str = "rgb_array",
    reward_mode: str = "normalized_dense",
    shadows: bool = True,
    print_sim_details: bool = True,
    record_video: bool = False,
    videos_dir: Optional[str] = None,
    max_steps_per_video: int = 500,
):
    """Set up ManiSkill environment with ImageProcessor,
    ConvertToTensor, and RecordEpisode wrappers."""
    env = gym.make(
        task_name,
        num_envs=num_envs,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        render_mode=render_mode,
        enable_shadow=shadows,
    )
    if obs_mode == "rgbd":
        env = ManiskillImageProcessorWrapper(
            env,
            camera_views=["base_camera"],
            camera_modes=["rgb"], # Other modes not currently supported
        )
    env = ConvertToTensorWrapper(env)
    if record_video:
        if videos_dir is None:
            raise ValueError('Arg "videos_dir" cannot be None if "record_video" is True.')
        env = RecordEpisode(
            env,
            videos_dir,
            trajectory_name=run_name,
            max_steps_per_video=max_steps_per_video,
        )
    if print_sim_details:
        env.unwrapped.print_sim_details()
        print(
            f"Observation space sample shape: {env.observation_space.sample().shape}"
        )
    return env
