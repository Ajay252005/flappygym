"""Unit tests for the FlappyBird Gymnasium environment."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import src.env
from src.env.flappy_env import FlappyBirdEnv


class TestFlappyBirdGymEnv:
    def test_gym_registration_and_make(self):
        env = gym.make("FlappyBird-v0")
        assert isinstance(env.unwrapped, FlappyBirdEnv)
        env.close()

    def test_gym_check_env(self):
        env = gym.make("FlappyBird-v0")
        # Should execute without errors or warnings
        check_env(env.unwrapped)
        env.close()

    def test_observation_space_contains(self):
        env = FlappyBirdEnv(normalize_obs=False)
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
        assert "score" in info

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, (float, int))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            if terminated or truncated:
                obs, info = env.reset()
        env.close()

    def test_normalized_observation_space(self):
        env = gym.make("FlappyBird-v0", normalize_obs=True)
        check_env(env.unwrapped)
        obs, _ = env.reset(seed=123)
        assert env.observation_space.contains(obs)
        # Normalized values should fall roughly within [-2, 2]
        assert np.all(obs >= -2.5) and np.all(obs <= 2.5)
        env.close()

    def test_reduced_observation_space(self):
        env = gym.make("FlappyBird-Reduced-v0")
        check_env(env.unwrapped)
        obs, _ = env.reset(seed=123)
        assert obs.shape == (4,)
        assert env.observation_space.shape == (4,)
        env.close()

    def test_sparse_reward_mode(self):
        env = gym.make("FlappyBird-Sparse-v0")
        check_env(env.unwrapped)
        obs, _ = env.reset(seed=999)
        # While bird hasn't passed pipe, reward must be 0.0
        obs, reward, term, trunc, info = env.step(0)
        if not term and info["score"] == 0:
            assert reward == 0.0
        env.close()

    def test_dense_reward_mode(self):
        env = FlappyBirdEnv(reward_mode="dense")
        obs, _ = env.reset(seed=999)
        # Step once with idle: frame survived should yield positive survival reward
        obs, reward, term, trunc, info = env.step(0)
        if not term:
            assert reward > 0.0  # survival reward
        env.close()

    def test_rgb_array_render(self):
        env = FlappyBirdEnv(render_mode="rgb_array")
        obs, _ = env.reset(seed=42)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (600, 400, 3)
        assert frame.dtype == np.uint8
        env.close()

    def test_random_agent_multi_episodes(self):
        """Run random actions across several episodes to guarantee robustness."""
        env = FlappyBirdEnv()
        obs, _ = env.reset(seed=42)
        total_steps = 0
        episodes = 0

        while total_steps < 1000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            if terminated or truncated:
                episodes += 1
                obs, _ = env.reset()

        assert episodes > 0
        assert total_steps >= 1000
        env.close()

