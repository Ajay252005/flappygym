"""Environment module and Gymnasium registration for Flappy Bird."""

from gymnasium.envs.registration import register
from src.env.flappy_env import FlappyBirdEnv

# Register default environment
register(
    id="FlappyBird-v0",
    entry_point="src.env:FlappyBirdEnv",
    max_episode_steps=None,
)

# Register variant with reduced observation (ablation)
register(
    id="FlappyBird-Reduced-v0",
    entry_point="src.env:FlappyBirdEnv",
    kwargs={"reduced_observation": True},
    max_episode_steps=None,
)

# Register variant with sparse reward (ablation)
register(
    id="FlappyBird-Sparse-v0",
    entry_point="src.env:FlappyBirdEnv",
    kwargs={"reward_mode": "sparse"},
    max_episode_steps=None,
)

__all__ = ["FlappyBirdEnv"]

