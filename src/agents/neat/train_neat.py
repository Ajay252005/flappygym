"""Training script for NEAT (Neuroevolution of Augmenting Topologies) agent."""

from typing import Any
import os
import sys
import copy
import json
import time
import random
import pickle
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import neat
import numpy as np

import src.env
from src.env.flappy_env import FlappyBirdEnv


def eval_single_genome(net: neat.nn.FeedForwardNetwork, seed: int = 42, max_pipes: int = 60) -> tuple[float, int, int]:
    """Evaluate a single neural network on FlappyBird.

    Returns:
        (fitness, score, frames_survived)
    """
    env = FlappyBirdEnv(normalize_obs=True)
    obs, info = env.reset(seed=seed)

    fitness = 0.0
    pipes_passed = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Feedforward activation
        output = net.activate(obs)
        action = 1 if output[0] > 0.0 else 0

        obs, reward, terminated, truncated, info = env.step(action)

        current_score = int(info["score"])

        # Heavily reward passing pipes (the primary objective)
        if current_score > pipes_passed:
            fitness += 200.0 * (current_score - pipes_passed)
            pipes_passed = current_score

        # Survival bonus (small, secondary)
        fitness += 0.5

        if info["score"] >= max_pipes:
            fitness += 2000.0
            break

    score = int(info["score"])
    frames = int(info["frames_survived"])
    env.close()

    return fitness, score, frames


class MetricsReporter(neat.reporting.BaseReporter):
    """Custom reporter to track training curve metrics per generation and cumulative environment steps."""

    def __init__(self):
        self.metrics = []
        self.gen_start = 0.0
        self.generation = 0
        self.best_score_ever = 0
        self.best_fitness_ever = -float("inf")
        self.best_genome_ever = None
        self.total_env_steps = 0
        self.gen_env_steps = 0

    def start_generation(self, generation: int):
        self.generation = generation
        self.gen_start = time.perf_counter()
        self.gen_env_steps = 0

    def add_steps(self, steps: int):
        self.gen_env_steps += steps
        self.total_env_steps += steps

    def post_evaluate(self, config, population, species, best_genome):
        elapsed = time.perf_counter() - self.gen_start
        fits = [g.fitness for g in population.values() if g.fitness is not None]
        mean_fit = float(np.mean(fits)) if fits else 0.0
        best_fit = float(best_genome.fitness)

        # Evaluate best genome to get actual pipe score
        best_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        _, best_score, eval_steps = eval_single_genome(best_net, seed=100 + self.generation, max_pipes=70)
        self.add_steps(eval_steps)

        if best_score > self.best_score_ever or (
            best_score == self.best_score_ever and best_fit > self.best_fitness_ever
        ):
            self.best_score_ever = max(self.best_score_ever, best_score)
            self.best_fitness_ever = max(self.best_fitness_ever, best_fit)
            self.best_genome_ever = copy.deepcopy(best_genome)
        elif self.best_genome_ever is None:
            self.best_genome_ever = copy.deepcopy(best_genome)

        record = {
            "generation": self.generation,
            "total_env_steps": self.total_env_steps,
            "gen_env_steps": self.gen_env_steps,
            "best_fitness": best_fit,
            "mean_fitness": mean_fit,
            "best_score": best_score,
            "best_score_ever": self.best_score_ever,
            "species_count": len(species.species),
            "time_sec": elapsed,
        }
        self.metrics.append(record)

        print(
            f"Gen {self.generation:3d} | Steps: {self.total_env_steps:8,d} | Best Fit: {best_fit:7.1f} | "
            f"Mean Fit: {mean_fit:6.1f} | Best Score: {best_score:3d} (Record: {self.best_score_ever:3d}) | "
            f"Time: {elapsed:.2f}s"
        )


def train_neat(
    config_path: str = "config/neat_config.txt",
    generations: int = 60,
    target_score: int = 50,
    seed: int = 42,
    save_model_path: str = "models/neat_best.pkl",
    metrics_path: str = "logs/neat_metrics.json",
) -> tuple[Any, dict]:
    """Train NEAT population to solve Flappy Bird."""
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = neat.Population(config)

    # Add standard reporter and our custom metrics reporter
    reporter = MetricsReporter()
    pop.add_reporter(reporter)

    def eval_genomes(genomes, cfg):
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, cfg)
            # Evaluate on 2 seeds for stability
            fit1, _, steps1 = eval_single_genome(net, seed=42, max_pipes=target_score + 10)
            fit2, _, steps2 = eval_single_genome(net, seed=1337, max_pipes=target_score + 10)
            genome.fitness = (fit1 + fit2) / 2.0
            reporter.add_steps(steps1 + steps2)

    print("=" * 65)
    print(" STARTING NEAT TRAINING (CHECKPOINT 3)")
    print(f" Target Score: {target_score} pipes | Population: {config.pop_size} | Max Generations: {generations}")
    print("=" * 65)

    start_total = time.perf_counter()
    winner = pop.run(eval_genomes, generations)
    total_time = time.perf_counter() - start_total

    print("\n" + "=" * 65)
    print(" NEAT TRAINING COMPLETE")
    print(f" Total Time: {total_time:.2f}s | Final Record Score: {reporter.best_score_ever}")
    print("=" * 65)

    # Save best genome (prefer highest-scoring genome tracked across generations)
    best_to_save = reporter.best_genome_ever if reporter.best_genome_ever is not None else winner
    with open(save_model_path, "wb") as f:
        pickle.dump(best_to_save, f)
    print(f" Saved best genome to: {save_model_path}")

    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "algorithm": "NEAT",
                "generations": len(reporter.metrics),
                "total_env_steps": reporter.total_env_steps,
                "total_time_seconds": total_time,
                "history": reporter.metrics,
            },
            f,
            indent=2,
        )
    print(f" Saved metrics to: {metrics_path}")

    return winner, config


def main():
    parser = argparse.ArgumentParser(description="Train NEAT agent on Flappy Bird")
    parser.add_argument("--config", type=str, default="config/neat_config.txt", help="Path to NEAT config file")
    parser.add_argument("--generations", type=int, default=60, help="Maximum number of generations (default: 60)")
    parser.add_argument("--target", type=int, default=50, help="Target pipes passed score (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="models/neat_best.pkl", help="Output path for best genome")
    parser.add_argument("--metrics", type=str, default="logs/neat_metrics.json", help="Output path for training metrics")

    args = parser.parse_args()

    train_neat(
        config_path=args.config,
        generations=args.generations,
        target_score=args.target,
        seed=args.seed,
        save_model_path=args.output,
        metrics_path=args.metrics,
    )


if __name__ == "__main__":
    main()
