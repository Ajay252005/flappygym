"""Integration tests verifying trained agent loading, inference, and action spaces."""

import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import neat
import pytest
from stable_baselines3 import DQN, PPO

import src.env
from src.env.flappy_env import FlappyBirdEnv


class TestAgentInference:
    def test_neat_model_inference(self):
        model_path = "models/neat_best.pkl"
        config_path = "config/neat_config.txt"
        assert os.path.exists(model_path), f"{model_path} missing"
        assert os.path.exists(config_path), f"{config_path} missing"

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        with open(model_path, "rb") as f:
            winner = pickle.load(f)

        net = neat.nn.FeedForwardNetwork.create(winner, config)

        env = FlappyBirdEnv(normalize_obs=True)
        obs, info = env.reset(seed=42)

        output = net.activate(obs)
        assert len(output) == 1
        action = 1 if output[0] > 0.0 else 0
        assert action in (0, 1)

        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        env.close()

    def test_dqn_model_inference(self):
        model_path = "models/dqn_best.zip"
        assert os.path.exists(model_path), f"{model_path} missing"

        model = DQN.load(model_path)
        env = FlappyBirdEnv(normalize_obs=True)
        obs, info = env.reset(seed=42)

        action, _ = model.predict(obs, deterministic=True)
        assert int(action) in (0, 1)

        obs, reward, terminated, truncated, info = env.step(int(action))
        assert env.observation_space.contains(obs)
        env.close()

    def test_ppo_model_inference(self):
        model_path = "models/ppo_best.zip"
        assert os.path.exists(model_path), f"{model_path} missing"

        model = PPO.load(model_path)
        env = FlappyBirdEnv(normalize_obs=True)
        obs, info = env.reset(seed=42)

        action, _ = model.predict(obs, deterministic=True)
        assert int(action) in (0, 1)

        obs, reward, terminated, truncated, info = env.step(int(action))
        assert env.observation_space.contains(obs)
        env.close()

