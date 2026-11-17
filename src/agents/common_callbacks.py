"""Shared Stable-Baselines3 callbacks for experiment tracking and checkpointing."""

import os
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class FlappyTrainingCallback(BaseCallback):
    """Tracks episode metrics across single or vectorized environments (n_envs >= 1),

    updates rolling scores, and checkpoints the best model based on rolling mean score.
    """

    def __init__(
        self,
        check_freq: int = 5000,
        save_path: str = "models/best_model.zip",
        metrics_path: str = "logs/metrics.json",
        window_size: int = 50,
        eval_env=None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.metrics_path = metrics_path
        self.window_size = window_size
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.episode_scores = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.history = []

        self.best_rolling_score = -float("inf")
        self.best_single_score = 0
        self.best_eval_score = -float("inf")

        self.ep_rewards = None
        self.ep_lengths = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs if self.training_env is not None else 1
        self.ep_rewards = np.zeros(n_envs, dtype=np.float64)
        self.ep_lengths = np.zeros(n_envs, dtype=np.int64)

    def _on_step(self) -> bool:
        rewards = np.array(self.locals["rewards"], dtype=np.float64)
        dones = np.array(self.locals["dones"], dtype=bool)
        infos = self.locals["infos"]
        n_envs = len(dones)

        if self.ep_rewards is None or len(self.ep_rewards) != n_envs:
            self.ep_rewards = np.zeros(n_envs, dtype=np.float64)
            self.ep_lengths = np.zeros(n_envs, dtype=np.int64)

        self.ep_rewards += rewards
        self.ep_lengths += 1

        for i in range(n_envs):
            if dones[i]:
                # Extract score from Monitor or FlappyBirdEnv info
                info = infos[i]
                score = int(info.get("score", 0))

                self.episode_scores.append(score)
                self.episode_rewards.append(float(self.ep_rewards[i]))
                self.episode_lengths.append(int(self.ep_lengths[i]))

                if score > self.best_single_score:
                    self.best_single_score = score

                if self.logger is not None:
                    self.logger.record("rollout/score", score)
                    self.logger.record("rollout/best_single_score", self.best_single_score)
                    self.logger.record("rollout/ep_length", int(self.ep_lengths[i]))

                self.ep_rewards[i] = 0.0
                self.ep_lengths[i] = 0

        # Periodic checkpointing and progress logging (aligned on environment timesteps)
        call_freq = max(1, self.check_freq // n_envs)
        if self.n_calls % call_freq == 0 and len(self.episode_scores) > 0:
            recent_scores = self.episode_scores[-self.window_size :]
            mean_score = float(np.mean(recent_scores))
            mean_reward = float(np.mean(self.episode_rewards[-self.window_size :]))
            mean_length = float(np.mean(self.episode_lengths[-self.window_size :]))

            record = {
                "step": int(self.num_timesteps),
                "episodes": len(self.episode_scores),
                "mean_score": mean_score,
                "best_single_score": int(self.best_single_score),
                "mean_reward": mean_reward,
                "mean_length": mean_length,
            }
            self.history.append(record)

            if self.logger is not None:
                self.logger.record("rollout/rolling_score_mean", mean_score)
                self.logger.record("rollout/rolling_reward_mean", mean_reward)
                self.logger.record("rollout/rolling_length_mean", mean_length)

            # Check for best model based on rolling mean score
            if mean_score > self.best_rolling_score:
                self.best_rolling_score = mean_score
                if self.save_path:
                    os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
                    self.model.save(self.save_path)
                    if self.verbose > 0:
                        print(
                            f"  [Step {self.num_timesteps:7d}] New best rolling mean score: {mean_score:.2f} "
                            f"(Max: {self.best_single_score:d}). Model saved to {self.save_path}"
                        )

            if self.verbose > 0 and self.n_calls % (call_freq * 2) == 0:
                print(
                    f"Step {self.num_timesteps:7d} | Episodes: {len(self.episode_scores):5d} | "
                    f"Mean Score (last {len(recent_scores)}): {mean_score:5.2f} | "
                    f"Max Score: {self.best_single_score:3d} | "
                    f"Mean Len: {mean_length:5.1f}"
                )

        return True

    def _on_training_end(self) -> None:
        """Save training history metrics to JSON on completion."""
        if self.best_rolling_score == -float("inf"):
            if len(self.episode_scores) > 0:
                self.best_rolling_score = float(np.mean(self.episode_scores[-self.window_size :]))
            else:
                self.best_rolling_score = 0.0
            if self.save_path:
                os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
                self.model.save(self.save_path)

        if self.metrics_path:
            os.makedirs(os.path.dirname(self.metrics_path) or ".", exist_ok=True)
            with open(self.metrics_path, "w") as f:
                json.dump(
                    {
                        "total_steps": int(self.num_timesteps),
                        "total_episodes": len(self.episode_scores),
                        "best_single_score": int(self.best_single_score),
                        "best_rolling_score": float(self.best_rolling_score),
                        "history": self.history,
                    },
                    f,
                    indent=2,
                )
            if self.verbose > 0:
                print(f"Training metrics saved to: {self.metrics_path}")
