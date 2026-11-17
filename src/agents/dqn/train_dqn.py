"""Training script for Deep Q-Network (DQN) agent on Flappy Bird (Extended Training)."""

import os
import sys
import time
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

import src.env
from src.env.flappy_env import FlappyBirdEnv
from src.agents.common_callbacks import FlappyTrainingCallback


def train_dqn(
    total_timesteps: int = 1_000_000,
    seed: int = 42,
    lr: float = 2.5e-4,
    buffer_size: int = 100_000,
    learning_starts: int = 10_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    target_update_interval: int = 1000,
    train_freq: int = 4,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.02,
    reward_mode: str = "dense",
    reduced_observation: bool = False,
    save_model_path: str = "models/dqn_best.zip",
    metrics_path: str = "logs/dqn_metrics.json",
    tb_log_dir: str = "logs/tensorboard/dqn",
) -> DQN:
    """Train DQN agent over extended timesteps (1M+ steps) with target network updates."""
    print("=" * 70)
    print(" STARTING EXTENDED DQN TRAINING (CHECKPOINT 4)")
    print(f" Timesteps: {total_timesteps:,} | Seed: {seed} | LR: {lr} | Replay Buffer: {buffer_size:,}")
    print(f" Learning Starts: {learning_starts:,} | Target Update Interval: {target_update_interval}")
    print(f" Exploration: 1.0 -> {exploration_final_eps} over {exploration_fraction*100:.0f}% of run")
    print(f" Reward Mode: {reward_mode} | Observation: {'Reduced' if reduced_observation else 'Full'}")
    print("=" * 70)

    os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    raw_env = FlappyBirdEnv(
        reward_mode=reward_mode,
        reduced_observation=reduced_observation,
        normalize_obs=True,
    )
    env = Monitor(raw_env)

    policy_kwargs = dict(
        net_arch=[128, 128],
        activation_fn=torch.nn.ReLU,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=lr,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_log_dir,
        seed=seed,
        verbose=0,
    )

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
    print(" DQN EXTENDED TRAINING COMPLETE")
    print(f" Elapsed Time: {elapsed:.2f}s ({elapsed / 60:.2f} min) | Speed: {total_timesteps / elapsed:,.1f} steps/s")
    print(f" Best Rolling Mean Score: {callback.best_rolling_score:.2f} | Record Score: {callback.best_single_score}")
    print(f" Best Model: {save_model_path}")
    print(f" Final Model: {final_path}")
    print(f" TensorBoard Logs: {tb_log_dir}")
    print("=" * 70)

    env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent on Flappy Bird (Extended)")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps (default: 1,000,000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate (default: 2.5e-4)")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--learning-starts", type=int, default=10000, help="Steps before learning starts")
    parser.add_argument("--target-interval", type=int, default=1000, help="Target network update interval")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--reward-mode", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument("--reduced", action="store_true", help="Use reduced observation space (no velocity)")
    parser.add_argument("--output", type=str, default="models/dqn_best.zip", help="Path to save best model")
    parser.add_argument("--metrics", type=str, default="logs/dqn_metrics.json", help="Path to save metrics")
    parser.add_argument("--tb-log", type=str, default="logs/tensorboard/dqn", help="TensorBoard log dir")

    args = parser.parse_args()

    train_dqn(
        total_timesteps=args.timesteps,
        seed=args.seed,
        lr=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        target_update_interval=args.target_interval,
        batch_size=args.batch_size,
        reward_mode=args.reward_mode,
        reduced_observation=args.reduced,
        save_model_path=args.output,
        metrics_path=args.metrics,
        tb_log_dir=args.tb_log,
    )


if __name__ == "__main__":
    main()
