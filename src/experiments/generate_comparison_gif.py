"""Generate a 4-panel side-by-side comparison GIF of Untrained (Random) vs Trained (NEAT, DQN, PPO) agents."""

import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import neat
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import DQN, PPO

import src.env
from src.env.flappy_env import FlappyBirdEnv


def generate_side_by_side_gif(
    output_path: str = "docs/untrained_vs_trained_comparison.gif",
    num_frames: int = 220,
    seed: int = 1003,
):
    """Render 4 synchronized Flappy Bird environments side-by-side into an animated GIF."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Load trained models
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config/neat_config.txt",
    )
    with open("models/neat_best.pkl", "rb") as f:
        winner = pickle.load(f)
    neat_net = neat.nn.FeedForwardNetwork.create(winner, config)

    dqn_model = DQN.load("models/dqn_best.zip")
    ppo_model = PPO.load("models/ppo_best.zip")

    agents = [
        ("Untrained (Random)", "random", (231, 76, 60)),
        ("NEAT (Neuroevolution)", "neat", (39, 174, 96)),
        ("DQN (Value-Based DRL)", "dqn", (230, 126, 34)),
        ("PPO (Policy Gradient DRL)", "ppo", (41, 128, 185)),
    ]

    envs = []
    obs_list = []
    done_list = []
    last_frames = []

    for _ in agents:
        env = FlappyBirdEnv(render_mode="rgb_array", normalize_obs=True)
        obs, _ = env.reset(seed=seed)
        envs.append(env)
        obs_list.append(obs)
        done_list.append(False)
        last_frames.append(env.render())

    rng = np.random.default_rng(seed)
    panel_w, panel_h = 220, 330
    header_h = 44
    pad = 8
    total_w = len(agents) * panel_w + (len(agents) + 1) * pad
    total_h = panel_h + header_h + 2 * pad

    composite_frames = []

    for step in range(num_frames):
        for i, (label, kind, _) in enumerate(agents):
            if not done_list[i]:
                obs = obs_list[i]
                if kind == "random":
                    action = int(rng.integers(0, 2))
                elif kind == "neat":
                    out = neat_net.activate(obs)
                    action = 1 if out[0] > 0.0 else 0
                elif kind == "dqn":
                    act, _ = dqn_model.predict(obs, deterministic=True)
                    action = int(act)
                else:
                    act, _ = ppo_model.predict(obs, deterministic=True)
                    action = int(act)

                obs, _, terminated, truncated, _ = envs[i].step(action)
                obs_list[i] = obs
                last_frames[i] = envs[i].render()
                if terminated or truncated:
                    done_list[i] = True

        # Sample every 2nd frame for smooth 30fps GIF with compact file size
        if step % 2 == 0:
            canvas = Image.new("RGB", (total_w, total_h), (28, 33, 40))
            draw = ImageDraw.Draw(canvas)

            for i, (label, _, badge_color) in enumerate(agents):
                x0 = pad + i * (panel_w + pad)
                y0 = pad

                # Draw header banner
                draw.rectangle([x0, y0, x0 + panel_w, y0 + header_h - 4], fill=badge_color)
                score = envs[i].game.score
                status = "CRASHED" if done_list[i] else f"Score: {score}"
                draw.text((x0 + 8, y0 + 5), label, fill=(255, 255, 255))
                draw.text((x0 + 8, y0 + 22), f"{status} | Frame: {envs[i].game.frames_survived}", fill=(245, 245, 245))

                # Resize and paste game viewport
                panel_img = Image.fromarray(last_frames[i]).resize((panel_w, panel_h), Image.Resampling.BILINEAR)
                canvas.paste(panel_img, (x0, y0 + header_h))

            composite_frames.append(canvas)

    for env in envs:
        env.close()

    composite_frames[0].save(
        output_path,
        save_all=True,
        append_images=composite_frames[1:],
        duration=40,
        loop=0,
        optimize=True,
    )
    print(f"Saved 4-panel comparison GIF to: {output_path} ({len(composite_frames)} frames)")


if __name__ == "__main__":
    generate_side_by_side_gif()

