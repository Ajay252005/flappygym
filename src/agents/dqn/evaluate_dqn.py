"""Evaluation and demo generation script for trained DQN agent."""

import os
import sys
import json
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import numpy as np
from PIL import Image
from stable_baselines3 import DQN

import src.env
from src.env.flappy_env import FlappyBirdEnv


def evaluate_dqn_agent(
    model_path: str = "models/dqn_best.zip",
    episodes: int = 30,
    max_pipes_per_ep: int = 100,
    save_gif_path: str = "docs/dqn_trained_demo.gif",
    render_human: bool = False,
    reduced_observation: bool = False,
) -> dict:
    """Evaluate trained DQN model and save demo recording."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Train first using train_dqn.py!")

    model = DQN.load(model_path)

    render_mode = "human" if render_human else "rgb_array"
    env = FlappyBirdEnv(
        render_mode=render_mode,
        reduced_observation=reduced_observation,
        normalize_obs=True,
    )

    scores = []
    frames_survived = []
    gif_frames = []

    print("=" * 60)
    print(" EVALUATING TRAINED DQN AGENT")
    print(f" Model: {model_path} | Target Episodes: {episodes}")
    print("=" * 60)

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=2000 + ep * 43)
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))

                if not render_human and ep == 0 and len(gif_frames) < 350:
                    frame = env.render()
                    if frame is not None:
                        gif_frames.append(Image.fromarray(frame))

                if info["score"] >= max_pipes_per_ep:
                    break

            score = info["score"]
            steps = info["frames_survived"]
            scores.append(score)
            frames_survived.append(steps)

            print(f"  Episode {ep + 1:2d}/{episodes} | Score: {score:3d} pipes | Frames Survived: {steps:5d}")
    except KeyboardInterrupt:
        print("\n  [Stopped by user] Closing window and summarizing completed episodes...")
        if not scores:
            scores.append(int(env.game.score))
            frames_survived.append(int(env.game.frames_survived))
    finally:
        env.close()

    avg_score = float(np.mean(scores))
    max_score = int(np.max(scores))
    min_score = int(np.min(scores))
    std_score = float(np.std(scores))

    print("=" * 60)
    print(" DQN EVALUATION SUMMARY")
    print(f"  Average Score : {avg_score:.2f} ± {std_score:.2f} pipes")
    print(f"  Max Score     : {max_score} pipes")
    print(f"  Min Score     : {min_score} pipes")
    print("=" * 60)

    if gif_frames and save_gif_path:
        os.makedirs(os.path.dirname(save_gif_path) or ".", exist_ok=True)
        gif_frames[0].save(
            save_gif_path,
            save_all=True,
            append_images=gif_frames[1::2],
            duration=33,
            loop=0,
        )
        print(f" Saved demo GIF to: {save_gif_path}")

    result = {
        "scores": [int(s) for s in scores],
        "avg_score": avg_score,
        "max_score": max_score,
        "min_score": min_score,
        "std_score": std_score,
    }
    if episodes >= 30 and not render_human:
        eval_json_path = "logs/dqn_eval.json"
        os.makedirs(os.path.dirname(eval_json_path) or ".", exist_ok=True)
        with open(eval_json_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN agent on Flappy Bird")
    parser.add_argument("--model", type=str, default="models/dqn_best.zip", help="Path to trained DQN model")
    parser.add_argument("--episodes", type=int, default=30, help="Number of test episodes (default: 30)")
    parser.add_argument("--max-pipes", type=int, default=100, help="Max pipes per episode to evaluate")
    parser.add_argument("--gif", type=str, default="docs/dqn_trained_demo.gif", help="Output GIF path")
    parser.add_argument("--reduced", action="store_true", help="Use reduced observation space (no velocity)")
    parser.add_argument("--render", action="store_true", help="Render visually in Pygame window")

    args = parser.parse_args()

    evaluate_dqn_agent(
        model_path=args.model,
        episodes=args.episodes,
        max_pipes_per_ep=args.max_pipes,
        save_gif_path=args.gif,
        render_human=args.render,
        reduced_observation=args.reduced,
    )


if __name__ == "__main__":
    main()

