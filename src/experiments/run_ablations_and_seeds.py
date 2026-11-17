"""Checkpoint 6: Multi-seed stability, reward-shaping ablation, observation-space ablation,
and zero-shot difficulty generalization experiments.

Runs controlled experiments across NEAT, DQN, and PPO, evaluates each configuration over
30 deterministic episodes, and exports structured data to `logs/checkpoint6_experiments.json`.
"""

import os
import sys
import json
import time
import copy
import random
import pickle
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import neat
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

import src.env
from src.env.flappy_env import FlappyBirdEnv
from src.agents.common_callbacks import FlappyTrainingCallback
from src.agents.neat.train_neat import train_neat
from src.agents.neat.evaluate_neat import evaluate_neat_agent
from src.agents.dqn.train_dqn import train_dqn
from src.agents.dqn.evaluate_dqn import evaluate_dqn_agent
from src.agents.ppo.train_ppo import train_ppo
from src.agents.ppo.evaluate_ppo import evaluate_ppo_agent


def evaluate_sb3_on_env(
    model,
    episodes: int = 30,
    max_pipes: int = 100,
    reward_mode: str = "dense",
    reduced_observation: bool = False,
    pipe_gap_size: int = 140,
    pipe_speed: float = 3.5,
    seed_offset: int = 5000,
) -> dict:
    """Evaluate a trained SB3 model (DQN or PPO) over `episodes` deterministic rollouts."""
    env = FlappyBirdEnv(
        reward_mode=reward_mode,
        reduced_observation=reduced_observation,
        normalize_obs=True,
        pipe_gap_size=pipe_gap_size,
        pipe_speed=pipe_speed,
    )
    scores = []
    lengths = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed_offset + ep * 41)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            if info["score"] >= max_pipes:
                break
        scores.append(int(info["score"]))
        lengths.append(int(info["frames_survived"]))

    env.close()
    return {
        "avg_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "avg_length": float(np.mean(lengths)),
        "scores": scores,
    }


def evaluate_neat_on_env(
    genome_path: str,
    config_path: str = "config/neat_config.txt",
    episodes: int = 30,
    max_pipes: int = 100,
    reduced_observation: bool = False,
    pipe_gap_size: int = 140,
    pipe_speed: float = 3.5,
    seed_offset: int = 5000,
) -> dict:
    """Evaluate a trained NEAT genome over `episodes` deterministic rollouts."""
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    with open(genome_path, "rb") as f:
        winner = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    env = FlappyBirdEnv(
        reduced_observation=reduced_observation,
        normalize_obs=True,
        pipe_gap_size=pipe_gap_size,
        pipe_speed=pipe_speed,
    )
    scores = []
    lengths = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed_offset + ep * 41)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            # If reduced_observation (4D) is passed to a 4-input net, or 5D to 5-input net
            output = net.activate(obs)
            action = 1 if output[0] > 0.0 else 0
            obs, _, terminated, truncated, info = env.step(action)
            if info["score"] >= max_pipes:
                break
        scores.append(int(info["score"]))
        lengths.append(int(info["frames_survived"]))

    env.close()
    return {
        "avg_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "avg_length": float(np.mean(lengths)),
        "scores": scores,
    }


def run_neat_ablation_variant(
    seed: int = 42,
    generations: int = 45,
    reward_mode: str = "dense",
    reduced_observation: bool = False,
    save_prefix: str = "neat_ablation",
) -> dict:
    """Train a NEAT variant for reward or observation ablation."""
    random.seed(seed)
    np.random.seed(seed)

    # Create temporary config if reduced_observation (4 inputs instead of 5)
    base_config_path = "config/neat_config.txt"
    if reduced_observation:
        with open(base_config_path, "r") as f:
            cfg_text = f.read().replace("num_inputs              = 5", "num_inputs              = 4")
        config_path = f"config/neat_config_4d.txt"
        with open(config_path, "w") as f:
            f.write(cfg_text)
    else:
        config_path = base_config_path

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    pop = neat.Population(config)

    history = []
    total_steps = 0
    best_genome_ever = None
    best_fit_ever = -float("inf")

    def eval_genome_custom(net, s: int):
        env = FlappyBirdEnv(
            reward_mode=reward_mode,
            reduced_observation=reduced_observation,
            normalize_obs=True,
        )
        obs, info = env.reset(seed=s)
        fitness = 0.0
        pipes_passed = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            output = net.activate(obs)
            action = 1 if output[0] > 0.0 else 0
            obs, _, terminated, truncated, info = env.step(action)
            cur = int(info["score"])
            if cur > pipes_passed:
                fitness += 200.0 * (cur - pipes_passed)
                pipes_passed = cur
            if reward_mode == "dense":
                fitness += 0.5
            if cur >= 60:
                fitness += 2000.0
                break
        steps = int(info["frames_survived"])
        score = int(info["score"])
        env.close()
        return fitness, score, steps

    gen_idx = 0

    def eval_genomes(genomes, cfg):
        nonlocal total_steps, best_genome_ever, best_fit_ever, gen_idx
        t0 = time.perf_counter()
        fits = []
        gen_best_g = None
        gen_best_fit = -float("inf")
        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, cfg)
            f1, _, st1 = eval_genome_custom(net, 42)
            f2, _, st2 = eval_genome_custom(net, 1337)
            g.fitness = (f1 + f2) / 2.0
            total_steps += st1 + st2
            fits.append(g.fitness)
            if g.fitness > gen_best_fit:
                gen_best_fit = g.fitness
                gen_best_g = g

        if gen_best_g is not None and gen_best_fit > best_fit_ever:
            best_fit_ever = gen_best_fit
            best_genome_ever = copy.deepcopy(gen_best_g)

        best_net = neat.nn.FeedForwardNetwork.create(gen_best_g, cfg)
        _, b_score, ev_steps = eval_genome_custom(best_net, 100 + gen_idx)
        total_steps += ev_steps
        history.append(
            {
                "generation": gen_idx,
                "total_env_steps": total_steps,
                "best_score": int(b_score),
                "best_fitness": float(gen_best_fit),
                "mean_fitness": float(np.mean(fits)),
                "time_sec": time.perf_counter() - t0,
            }
        )
        gen_idx += 1

    t_start = time.perf_counter()
    winner = pop.run(eval_genomes, generations)
    elapsed = time.perf_counter() - t_start

    final_genome = best_genome_ever if best_genome_ever is not None else winner
    model_path = f"models/{save_prefix}.pkl"
    os.makedirs("models", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(final_genome, f)

    eval_res = evaluate_neat_on_env(
        genome_path=model_path,
        config_path=config_path,
        episodes=30,
        reduced_observation=reduced_observation,
    )
    return {
        "total_steps": total_steps,
        "training_time_sec": elapsed,
        "history": history,
        "eval": eval_res,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Checkpoint 6 Experiments & Ablations")
    parser.add_argument("--rl-steps", type=int, default=400_000, help="Timesteps for ablation/extra-seed DRL runs")
    parser.add_argument("--neat-gens", type=int, default=45, help="Generations for ablation/extra-seed NEAT runs")
    args = parser.parse_args()

    os.makedirs("logs/ablations", exist_ok=True)
    os.makedirs("models/ablations", exist_ok=True)

    results = {
        "multi_seed": {"NEAT": {}, "DQN": {}, "PPO": {}},
        "reward_ablation": {},
        "observation_ablation": {},
        "difficulty_generalization": {},
    }

    # =========================================================================
    # 1. LOAD EXISTING SEED 42 BASELINES
    # =========================================================================
    print("\n" + "=" * 75)
    print(" PHASE 1: LOADING SEED 42 BASELINES & RUNNING MULTI-SEED (SEEDS 42, 123, 456)")
    print("=" * 75)

    with open("logs/neat_metrics.json") as f:
        neat_s42_train = json.load(f)
    with open("logs/neat_eval.json") as f:
        neat_s42_eval = json.load(f)
    results["multi_seed"]["NEAT"]["42"] = {
        "total_steps": neat_s42_train["total_env_steps"],
        "training_time_sec": neat_s42_train["total_time_seconds"],
        "train_max_score": max(h["best_score_ever"] for h in neat_s42_train["history"]),
        "eval_avg": neat_s42_eval["avg_score"],
        "eval_std": neat_s42_eval["std_score"],
        "eval_max": neat_s42_eval["max_score"],
        "scores": neat_s42_eval["scores"],
        "history": [
            {"step": h["total_env_steps"], "score": h["best_score_ever"]}
            for h in neat_s42_train["history"]
        ],
    }

    with open("logs/dqn_metrics.json") as f:
        dqn_s42_train = json.load(f)
    with open("logs/dqn_eval.json") as f:
        dqn_s42_eval = json.load(f)
    results["multi_seed"]["DQN"]["42"] = {
        "total_steps": dqn_s42_train["total_steps"],
        "training_time_sec": 297.6,
        "train_max_score": dqn_s42_train["best_single_score"],
        "eval_avg": dqn_s42_eval["avg_score"],
        "eval_std": dqn_s42_eval["std_score"],
        "eval_max": dqn_s42_eval["max_score"],
        "scores": dqn_s42_eval["scores"],
        "history": [
            {"step": h["step"], "score": h["mean_score"]}
            for h in dqn_s42_train["history"]
        ],
    }

    with open("logs/ppo_metrics.json") as f:
        ppo_s42_train = json.load(f)
    with open("logs/ppo_eval.json") as f:
        ppo_s42_eval = json.load(f)
    results["multi_seed"]["PPO"]["42"] = {
        "total_steps": ppo_s42_train["total_steps"],
        "training_time_sec": 229.8,
        "train_max_score": ppo_s42_train["best_single_score"],
        "eval_avg": ppo_s42_eval["avg_score"],
        "eval_std": ppo_s42_eval["std_score"],
        "eval_max": ppo_s42_eval["max_score"],
        "scores": ppo_s42_eval["scores"],
        "history": [
            {"step": h["step"], "score": h["mean_score"]}
            for h in ppo_s42_train["history"]
        ],
    }

    # Run additional seeds 123 and 456 for NEAT
    for seed in [123, 456]:
        m_path = f"models/ablations/neat_seed{seed}.pkl"
        log_path = f"logs/ablations/neat_seed{seed}.json"
        t0 = time.perf_counter()
        train_neat(
            generations=args.neat_gens,
            target_score=50,
            seed=seed,
            save_model_path=m_path,
            metrics_path=log_path,
        )
        el = time.perf_counter() - t0
        ev = evaluate_neat_on_env(m_path, episodes=30)
        with open(log_path) as f:
            tr_data = json.load(f)
        results["multi_seed"]["NEAT"][str(seed)] = {
            "total_steps": tr_data["total_env_steps"],
            "training_time_sec": el,
            "train_max_score": max(h["best_score_ever"] for h in tr_data["history"]),
            "eval_avg": ev["avg_score"],
            "eval_std": ev["std_score"],
            "eval_max": ev["max_score"],
            "scores": ev["scores"],
            "history": [
                {"step": h["total_env_steps"], "score": h["best_score_ever"]}
                for h in tr_data["history"]
            ],
        }

    # Run additional seeds 123 and 456 for PPO
    for seed in [123, 456]:
        m_path = f"models/ablations/ppo_seed{seed}.zip"
        log_path = f"logs/ablations/ppo_seed{seed}.json"
        t0 = time.perf_counter()
        model = train_ppo(
            total_timesteps=args.rl_steps,
            n_envs=8,
            seed=seed,
            save_model_path=m_path,
            metrics_path=log_path,
            tb_log_dir=f"logs/tensorboard/ppo_seed{seed}",
        )
        el = time.perf_counter() - t0
        best_m = PPO.load(m_path) if os.path.exists(m_path) else model
        ev = evaluate_sb3_on_env(best_m, episodes=30)
        with open(log_path) as f:
            tr_data = json.load(f)
        results["multi_seed"]["PPO"][str(seed)] = {
            "total_steps": tr_data["total_steps"],
            "training_time_sec": el,
            "train_max_score": tr_data["best_single_score"],
            "eval_avg": ev["avg_score"],
            "eval_std": ev["std_score"],
            "eval_max": ev["max_score"],
            "scores": ev["scores"],
            "history": [
                {"step": h["step"], "score": h["mean_score"]}
                for h in tr_data["history"]
            ],
        }

    # Run additional seeds 123 and 456 for DQN
    for seed in [123, 456]:
        m_path = f"models/ablations/dqn_seed{seed}.zip"
        log_path = f"logs/ablations/dqn_seed{seed}.json"
        t0 = time.perf_counter()
        model = train_dqn(
            total_timesteps=args.rl_steps,
            seed=seed,
             exploration_fraction=0.35,
            save_model_path=m_path,
            metrics_path=log_path,
            tb_log_dir=f"logs/tensorboard/dqn_seed{seed}",
        )
        el = time.perf_counter() - t0
        best_m = DQN.load(m_path) if os.path.exists(m_path) else model
        ev = evaluate_sb3_on_env(best_m, episodes=30)
        with open(log_path) as f:
            tr_data = json.load(f)
        results["multi_seed"]["DQN"][str(seed)] = {
            "total_steps": tr_data["total_steps"],
            "training_time_sec": el,
            "train_max_score": tr_data["best_single_score"],
            "eval_avg": ev["avg_score"],
            "eval_std": ev["std_score"],
            "eval_max": ev["max_score"],
            "scores": ev["scores"],
            "history": [
                {"step": h["step"], "score": h["mean_score"]}
                for h in tr_data["history"]
            ],
        }

    # =========================================================================
    # 2. REWARD SHAPING ABLATION (DENSE vs SPARSE)
    # =========================================================================
    print("\n" + "=" * 75)
    print(" PHASE 2: REWARD SHAPING ABLATION (DENSE vs SPARSE REWARD)")
    print("=" * 75)

    # Dense baselines from Seed 42
    results["reward_ablation"]["PPO_Dense"] = results["multi_seed"]["PPO"]["42"]
    results["reward_ablation"]["DQN_Dense"] = results["multi_seed"]["DQN"]["42"]
    results["reward_ablation"]["NEAT_Dense"] = results["multi_seed"]["NEAT"]["42"]

    # Sparse PPO
    ppo_sparse_model_path = "models/ablations/ppo_sparse.zip"
    ppo_sparse_log_path = "logs/ablations/ppo_sparse.json"
    t0 = time.perf_counter()
    m_ppo_sparse = train_ppo(
        total_timesteps=args.rl_steps,
        n_envs=8,
        seed=42,
        reward_mode="sparse",
        save_model_path=ppo_sparse_model_path,
        metrics_path=ppo_sparse_log_path,
        tb_log_dir="logs/tensorboard/ppo_sparse",
    )
    el_ppo_sparse = time.perf_counter() - t0
    best_ppo_sparse = PPO.load(ppo_sparse_model_path) if os.path.exists(ppo_sparse_model_path) else m_ppo_sparse
    ev_ppo_sparse = evaluate_sb3_on_env(best_ppo_sparse, episodes=30, reward_mode="sparse")
    with open(ppo_sparse_log_path) as f:
        tr_ppo_sparse = json.load(f)
    results["reward_ablation"]["PPO_Sparse"] = {
        "total_steps": tr_ppo_sparse["total_steps"],
        "training_time_sec": el_ppo_sparse,
        "train_max_score": tr_ppo_sparse["best_single_score"],
        "eval_avg": ev_ppo_sparse["avg_score"],
        "eval_std": ev_ppo_sparse["std_score"],
        "eval_max": ev_ppo_sparse["max_score"],
        "history": [{"step": h["step"], "score": h["mean_score"]} for h in tr_ppo_sparse["history"]],
    }

    # Sparse DQN
    dqn_sparse_model_path = "models/ablations/dqn_sparse.zip"
    dqn_sparse_log_path = "logs/ablations/dqn_sparse.json"
    t0 = time.perf_counter()
    m_dqn_sparse = train_dqn(
        total_timesteps=args.rl_steps,
        seed=42,
        reward_mode="sparse",
        save_model_path=dqn_sparse_model_path,
        metrics_path=dqn_sparse_log_path,
        tb_log_dir="logs/tensorboard/dqn_sparse",
    )
    el_dqn_sparse = time.perf_counter() - t0
    best_dqn_sparse = DQN.load(dqn_sparse_model_path) if os.path.exists(dqn_sparse_model_path) else m_dqn_sparse
    ev_dqn_sparse = evaluate_sb3_on_env(best_dqn_sparse, episodes=30, reward_mode="sparse")
    with open(dqn_sparse_log_path) as f:
        tr_dqn_sparse = json.load(f)
    results["reward_ablation"]["DQN_Sparse"] = {
        "total_steps": tr_dqn_sparse["total_steps"],
        "training_time_sec": el_dqn_sparse,
        "train_max_score": tr_dqn_sparse["best_single_score"],
        "eval_avg": ev_dqn_sparse["avg_score"],
        "eval_std": ev_dqn_sparse["std_score"],
        "eval_max": ev_dqn_sparse["max_score"],
        "history": [{"step": h["step"], "score": h["mean_score"]} for h in tr_dqn_sparse["history"]],
    }

    # Sparse NEAT (no survival fitness bonus, only pipe passage)
    neat_sparse_res = run_neat_ablation_variant(
        seed=42,
        generations=args.neat_gens,
        reward_mode="sparse",
        reduced_observation=False,
        save_prefix="ablations/neat_sparse",
    )
    results["reward_ablation"]["NEAT_Sparse"] = {
        "total_steps": neat_sparse_res["total_steps"],
        "training_time_sec": neat_sparse_res["training_time_sec"],
        "train_max_score": max(h["best_score"] for h in neat_sparse_res["history"]),
        "eval_avg": neat_sparse_res["eval"]["avg_score"],
        "eval_std": neat_sparse_res["eval"]["std_score"],
        "eval_max": neat_sparse_res["eval"]["max_score"],
        "history": [{"step": h["total_env_steps"], "score": h["best_score"]} for h in neat_sparse_res["history"]],
    }

    # =========================================================================
    # 3. OBSERVATION SPACE ABLATION (FULL 5D vs REDUCED 4D WITHOUT VELOCITY)
    # =========================================================================
    print("\n" + "=" * 75)
    print(" PHASE 3: OBSERVATION ABLATION (FULL 5D vs REDUCED 4D NO-VELOCITY)")
    print("=" * 75)

    results["observation_ablation"]["PPO_Full5D"] = results["multi_seed"]["PPO"]["42"]
    results["observation_ablation"]["DQN_Full5D"] = results["multi_seed"]["DQN"]["42"]
    results["observation_ablation"]["NEAT_Full5D"] = results["multi_seed"]["NEAT"]["42"]

    # Reduced 4D PPO
    ppo_red_model_path = "models/ablations/ppo_reduced4d.zip"
    ppo_red_log_path = "logs/ablations/ppo_reduced4d.json"
    t0 = time.perf_counter()
    m_ppo_red = train_ppo(
        total_timesteps=args.rl_steps,
        n_envs=8,
        seed=42,
        reduced_observation=True,
        save_model_path=ppo_red_model_path,
        metrics_path=ppo_red_log_path,
        tb_log_dir="logs/tensorboard/ppo_reduced4d",
    )
    el_ppo_red = time.perf_counter() - t0
    best_ppo_red = PPO.load(ppo_red_model_path) if os.path.exists(ppo_red_model_path) else m_ppo_red
    ev_ppo_red = evaluate_sb3_on_env(best_ppo_red, episodes=30, reduced_observation=True)
    with open(ppo_red_log_path) as f:
        tr_ppo_red = json.load(f)
    results["observation_ablation"]["PPO_Reduced4D"] = {
        "total_steps": tr_ppo_red["total_steps"],
        "training_time_sec": el_ppo_red,
        "train_max_score": tr_ppo_red["best_single_score"],
        "eval_avg": ev_ppo_red["avg_score"],
        "eval_std": ev_ppo_red["std_score"],
        "eval_max": ev_ppo_red["max_score"],
        "history": [{"step": h["step"], "score": h["mean_score"]} for h in tr_ppo_red["history"]],
    }

    # Reduced 4D DQN
    dqn_red_model_path = "models/ablations/dqn_reduced4d.zip"
    dqn_red_log_path = "logs/ablations/dqn_reduced4d.json"
    t0 = time.perf_counter()
    m_dqn_red = train_dqn(
        total_timesteps=args.rl_steps,
        seed=42,
        reduced_observation=True,
        save_model_path=dqn_red_model_path,
        metrics_path=dqn_red_log_path,
        tb_log_dir="logs/tensorboard/dqn_reduced4d",
    )
    el_dqn_red = time.perf_counter() - t0
    best_dqn_red = DQN.load(dqn_red_model_path) if os.path.exists(dqn_red_model_path) else m_dqn_red
    ev_dqn_red = evaluate_sb3_on_env(best_dqn_red, episodes=30, reduced_observation=True)
    with open(dqn_red_log_path) as f:
        tr_dqn_red = json.load(f)
    results["observation_ablation"]["DQN_Reduced4D"] = {
        "total_steps": tr_dqn_red["total_steps"],
        "training_time_sec": el_dqn_red,
        "train_max_score": tr_dqn_red["best_single_score"],
        "eval_avg": ev_dqn_red["avg_score"],
        "eval_std": ev_dqn_red["std_score"],
        "eval_max": ev_dqn_red["max_score"],
        "history": [{"step": h["step"], "score": h["mean_score"]} for h in tr_dqn_red["history"]],
    }

    # Reduced 4D NEAT
    neat_red_res = run_neat_ablation_variant(
        seed=42,
        generations=args.neat_gens,
        reward_mode="dense",
        reduced_observation=True,
        save_prefix="ablations/neat_reduced4d",
    )
    results["observation_ablation"]["NEAT_Reduced4D"] = {
        "total_steps": neat_red_res["total_steps"],
        "training_time_sec": neat_red_res["training_time_sec"],
        "train_max_score": max(h["best_score"] for h in neat_red_res["history"]),
        "eval_avg": neat_red_res["eval"]["avg_score"],
        "eval_std": neat_red_res["eval"]["std_score"],
        "eval_max": neat_red_res["eval"]["max_score"],
        "history": [{"step": h["total_env_steps"], "score": h["best_score"]} for h in neat_red_res["history"]],
    }

    # =========================================================================
    # 4. ZERO-SHOT DIFFICULTY GENERALIZATION (EASY / STANDARD / HARD)
    # =========================================================================
    print("\n" + "=" * 75)
    print(" PHASE 4: ZERO-SHOT DIFFICULTY GENERALIZATION")
    print("=" * 75)

    difficulties = {
        "Easy (Gap=160, Speed=3.0)": {"gap": 160, "speed": 3.0},
        "Standard (Gap=140, Speed=3.5)": {"gap": 140, "speed": 3.5},
        "Hard (Gap=115, Speed=4.5)": {"gap": 115, "speed": 4.5},
    }

    ppo_main = PPO.load("models/ppo_best.zip")
    dqn_main = DQN.load("models/dqn_best.zip")

    for diff_label, params in difficulties.items():
        results["difficulty_generalization"][diff_label] = {
            "NEAT": evaluate_neat_on_env(
                "models/neat_best.pkl",
                episodes=30,
                pipe_gap_size=params["gap"],
                pipe_speed=params["speed"],
            ),
            "DQN": evaluate_sb3_on_env(
                dqn_main,
                episodes=30,
                pipe_gap_size=params["gap"],
                pipe_speed=params["speed"],
            ),
            "PPO": evaluate_sb3_on_env(
                ppo_main,
                episodes=30,
                pipe_gap_size=params["gap"],
                pipe_speed=params["speed"],
            ),
        }
        print(
            f"  {diff_label:<32} -> "
            f"NEAT: {results['difficulty_generalization'][diff_label]['NEAT']['avg_score']:.2f} | "
            f"DQN: {results['difficulty_generalization'][diff_label]['DQN']['avg_score']:.2f} | "
            f"PPO: {results['difficulty_generalization'][diff_label]['PPO']['avg_score']:.2f}"
        )

    # Compute multi-seed summary statistics
    summary_stats = {}
    for algo in ["NEAT", "DQN", "PPO"]:
        seed_avgs = [v["eval_avg"] for v in results["multi_seed"][algo].values()]
        seed_maxs = [v["eval_max"] for v in results["multi_seed"][algo].values()]
        seed_times = [v["training_time_sec"] for v in results["multi_seed"][algo].values()]
        seed_steps = [v["total_steps"] for v in results["multi_seed"][algo].values()]
        summary_stats[algo] = {
            "mean_across_seeds": float(np.mean(seed_avgs)),
            "std_across_seeds": float(np.std(seed_avgs)),
            "max_across_seeds": int(np.max(seed_maxs)),
            "mean_train_time_sec": float(np.mean(seed_times)),
            "mean_total_steps": int(np.mean(seed_steps)),
        }
    results["multi_seed_summary"] = summary_stats

    out_path = "logs/checkpoint6_experiments.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved all Checkpoint 6 experiment data to {out_path}")


if __name__ == "__main__":
    main()
