"""Training script for Proximal Policy Optimization (PPO) agent on Flappy Bird.

Configured with 8 parallel vectorized environments for sample efficiency and training stability.
"""

import os
import sys
import time
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import src.env
from src.env.flappy_env import FlappyBirdEnv
from src.agents.common_callbacks import FlappyTrainingCallback


def make_env_fn(reward_mode: str = "dense", reduced_observation: bool = False):
    def _init():
        return FlappyBirdEnv(
            reward_mode=reward_mode,
            reduced_observation=reduced_observation,
            normalize_obs=True,
        )
    return _init


def train_ppo(
    total_timesteps: int = 1_000_000,
    n_envs: int = 8,
    seed: int = 42,
    lr: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    reward_mode: str = "dense",
    reduced_observation: bool = False,
    save_model_path: str = "models/ppo_best.zip",
    metrics_path: str = "logs/ppo_metrics.json",
    tb_log_dir: str = "logs/tensorboard/ppo",
) -> PPO:
    """Train PPO agent using n_envs parallel environments over extended timesteps."""
    print("=" * 70)
    print(" STARTING EXTENDED PPO TRAINING (CHECKPOINT 5)")
    print(f" Timesteps: {total_timesteps:,} | Parallel Envs: {n_envs} | Seed: {seed}")
    print(f" LR: {lr} | n_steps per env: {n_steps} (Total batch: {n_steps * n_envs:,})")
    print(f" Reward Mode: {reward_mode} | Observation: {'Reduced' if reduced_observation else 'Full'}")
    print("=" * 70)

    os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # Create 8 parallel vectorized environments
    vec_env = make_vec_env(
        make_env_fn(reward_mode=reward_mode, reduced_observation=reduced_observation),
        n_envs=n_envs,
        seed=seed,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_log_dir,
        seed=seed,
        verbose=0,
    )

    # Checkpoint every 5,000 steps
    callback = FlappyTrainingCallback(
        check_freq=5000,
        save_path=save_model_path,
        metrics_path=metrics_path,
        window_size=100,
        verbose=1,
    )

    start_time = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    elapsed = time.perf_counter() - start_time

    final_path = save_model_path.replace(".zip", "_final.zip")
    model.save(final_path)

    print("\n" + "=" * 70)
    print(" PPO EXTENDED TRAINING COMPLETE")
    print(f" Elapsed Time: {elapsed:.2f}s ({elapsed / 60:.2f} min) | Speed: {total_timesteps / elapsed:,.1f} steps/s")
    print(f" Best Rolling Mean Score: {callback.best_rolling_score:.2f} | Record Score: {callback.best_single_score}")
    print(f" Best Model: {save_model_path}")
    print(f" Final Model: {final_path}")
    print(f" TensorBoard Logs: {tb_log_dir}")
    print("=" * 70)

    vec_env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Flappy Bird (Extended)")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps (default: 1,000,000)")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout steps per env")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--reward-mode", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument("--reduced", action="store_true", help="Use reduced observation space (no velocity)")
    parser.add_argument("--output", type=str, default="models/ppo_best.zip", help="Path to save best model")
    parser.add_argument("--metrics", type=str, default="logs/ppo_metrics.json", help="Path to save metrics")
    parser.add_argument("--tb-log", type=str, default="logs/tensorboard/ppo", help="TensorBoard log dir")

    args = parser.parse_args()

    train_ppo(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        reward_mode=args.reward_mode,
        reduced_observation=args.reduced,
        save_model_path=args.output,
        metrics_path=args.metrics,
        tb_log_dir=args.tb_log,
    )


if __name__ == "__main__":
    main()
