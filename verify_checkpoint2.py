"""Verification script for Checkpoint 2: Gymnasium Environment.

Validates:
  1. Gymnasium's official env_checker on multiple environment configurations.
  2. A random agent running 5,000+ steps across multiple episodes without errors.
  3. Visual verification by capturing and saving a short RGB GIF of random play.
"""

import os
import time
import argparse
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from PIL import Image

import src.env
from src.env.flappy_env import FlappyBirdEnv


def run_env_checker():
    print("=" * 60)
    print("STEP 1: Validating with Gymnasium's check_env()")
    print("=" * 60)

    configs = [
        ("Default FlappyBird-v0", lambda: gym.make("FlappyBird-v0")),
        ("Normalized Observations", lambda: gym.make("FlappyBird-v0", normalize_obs=True)),
        ("Reduced Observations (no velocity)", lambda: gym.make("FlappyBird-Reduced-v0")),
        ("Sparse Rewards", lambda: gym.make("FlappyBird-Sparse-v0")),
        ("Dense Rewards", lambda: gym.make("FlappyBird-v0", reward_mode="dense")),
    ]

    for label, env_fn in configs:
        env = env_fn()
        check_env(env.unwrapped)
        print(f" [PASS] {label}: check_env passed.")
        env.close()

    print("All Gymnasium env checks passed successfully!\n")


def run_random_agent(total_steps: int = 5000, seed: int = 42, record_gif: bool = True):
    print("=" * 60)
    print(f"STEP 2: Running Random Agent for {total_steps:,} steps")
    print("=" * 60)

    render_mode = "rgb_array" if record_gif else None
    env = FlappyBirdEnv(render_mode=render_mode)
    obs, info = env.reset(seed=seed)

    step_count = 0
    episodes = 0
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    current_reward = 0.0
    current_length = 0

    frames = []
    start_time = time.perf_counter()

    while step_count < total_steps:
        # Sample random action uniformly
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        current_length += 1
        current_reward += reward

        # Record frames from the first episode for demo
        if record_gif and episodes == 0 and len(frames) < 180:
            frame = env.render()
            if frame is not None:
                frames.append(Image.fromarray(frame))

        if terminated or truncated:
            episodes += 1
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
            episode_scores.append(info.get("score", 0))

            current_reward = 0.0
            current_length = 0
            obs, info = env.reset()

            if episodes % 25 == 0:
                print(
                    f"  Episode {episodes:3d} | Steps: {step_count:5d}/{total_steps} | "
                    f"Mean Length: {sum(episode_lengths[-25:]) / 25:.1f} | "
                    f"Max Score so far: {max(episode_scores)}"
                )

    elapsed = time.perf_counter() - start_time
    env.close()

    fps = step_count / elapsed
    avg_len = sum(episode_lengths) / max(1, len(episode_lengths))
    max_score = max(episode_scores) if episode_scores else 0
    avg_score = sum(episode_scores) / max(1, len(episode_scores))

    print("\n" + "=" * 60)
    print(" RANDOM AGENT BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Total Steps Executed : {step_count:,}")
    print(f"  Total Episodes       : {episodes}")
    print(f"  Execution Time       : {elapsed:.2f} seconds")
    print(f"  Simulation Speed     : {fps:,.1f} steps/second")
    print(f"  Average Episode Len  : {avg_len:.1f} steps")
    print(f"  Average Score        : {avg_score:.2f} pipes")
    print(f"  Max Score (Random)   : {max_score} pipes")
    print("=" * 60)

    if record_gif and frames:
        gif_path = "docs/random_agent_demo.gif"
        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1::2],
            duration=33,
            loop=0,
        )
        print(f"  Saved random agent demo GIF to: {gif_path}")

    print("\n [DONE] Random agent ran without any errors!")


def main():
    parser = argparse.ArgumentParser(description="Verify Checkpoint 2: Gymnasium Environment")
    parser.add_argument("--steps", type=int, default=5000, help="Number of random steps to run (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--no-gif", action="store_true", help="Disable saving random agent demo GIF")

    args = parser.parse_args()

    run_env_checker()
    run_random_agent(total_steps=args.steps, seed=args.seed, record_gif=not args.no_gif)


if __name__ == "__main__":
    main()

