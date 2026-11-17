"""Generate publication-quality figures and summary tables for Checkpoint 6 & 7:
1. Multi-Seed Stability (Seeds 42, 123, 456) across NEAT, DQN, and PPO
2. Reward-Shaping Ablation (Dense vs Sparse Reward)
3. Observation-Space Ablation (Full 5D vs Reduced 4D without vertical velocity)
4. Zero-Shot Difficulty Generalization (Easy, Standard, Hard pipe configurations)
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import matplotlib.pyplot as plt
import numpy as np


def generate_ablation_plots(
    input_json: str = "logs/checkpoint6_experiments.json",
    output_plot: str = "docs/ablation_studies.png",
):
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Experiment file not found: {input_json}. Run run_ablations_and_seeds.py first.")

    with open(input_json, "r") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_plot) or ".", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))
    fig.suptitle(
        "Checkpoint 6 & 7: Multi-Seed Stability, Reward & Observation Ablations, and Generalization",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    algos = ["NEAT", "DQN", "PPO"]
    palette = {"NEAT": "#27ae60", "DQN": "#e74c3c", "PPO": "#3498db"}

    # -------------------------------------------------------------------------
    # Panel 1 (Top-Left): Multi-Seed Stability (Seeds 42, 123, 456)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    seeds = ["42", "123", "456"]
    x = np.arange(len(algos))
    width = 0.24

    seed_colors = ["#2c3e50", "#16a085", "#8e44ad"]
    for idx, s in enumerate(seeds):
        vals = [data["multi_seed"][a][s]["eval_avg"] for a in algos]
        errs = [data["multi_seed"][a][s]["eval_std"] * 0.25 for a in algos]
        bars = ax1.bar(
            x + (idx - 1) * width,
            vals,
            width,
            yerr=errs,
            capsize=4,
            label=f"Seed {s}",
            color=seed_colors[idx],
            alpha=0.88,
            edgecolor="black",
        )
        for b in bars:
            h = b.get_height()
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                h + 1.5,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax1.set_title("1. Multi-Seed Stability (30-Ep Mean Score per Seed)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Pipes Passed (30 Episodes)", fontsize=11)
    ax1.set_ylim(0, max(75, ax1.get_ylim()[1] * 1.15))
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 2 (Top-Right): Reward Shaping Ablation (Dense vs Sparse)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    dense_vals = [data["reward_ablation"][f"{a}_Dense"]["eval_avg"] for a in algos]
    sparse_vals = [data["reward_ablation"][f"{a}_Sparse"]["eval_avg"] for a in algos]

    w2 = 0.34
    b_dense = ax2.bar(
        x - w2 / 2,
        dense_vals,
        w2,
        label="Dense Reward (+Survival & Centering)",
        color="#2980b9",
        edgecolor="black",
    )
    b_sparse = ax2.bar(
        x + w2 / 2,
        sparse_vals,
        w2,
        label="Sparse Reward (+1 per Pipe Only)",
        color="#e67e22",
        edgecolor="black",
    )

    for b in list(b_dense) + list(b_sparse):
        h = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            h + 1.0,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax2.set_title("2. Reward Shaping Ablation: Dense vs. Sparse", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Pipes Passed (30 Episodes)", fontsize=11)
    ax2.set_ylim(0, max(70, max(dense_vals) * 1.2))
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 3 (Bottom-Left): Observation Ablation (Full 5D vs Reduced 4D)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    full5d_vals = [data["observation_ablation"][f"{a}_Full5D"]["eval_avg"] for a in algos]
    red4d_vals = [data["observation_ablation"][f"{a}_Reduced4D"]["eval_avg"] for a in algos]

    b_5d = ax3.bar(
        x - w2 / 2,
        full5d_vals,
        w2,
        label="Full 5D State (Includes Bird Velocity vy)",
        color="#27ae60",
        edgecolor="black",
    )
    b_4d = ax3.bar(
        x + w2 / 2,
        red4d_vals,
        w2,
        label="Reduced 4D State (Omits Velocity vy)",
        color="#c0392b",
        edgecolor="black",
    )

    for b in list(b_5d) + list(b_4d):
        h = b.get_height()
        ax3.text(
            b.get_x() + b.get_width() / 2.0,
            h + 1.0,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax3.set_title("3. Observation Space Ablation: Full 5D vs. Reduced 4D", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(algos, fontsize=11, fontweight="bold")
    ax3.set_ylabel("Mean Pipes Passed (30 Episodes)", fontsize=11)
    ax3.set_ylim(0, max(70, max(full5d_vals) * 1.2))
    ax3.legend(loc="upper left", frameon=True)
    ax3.grid(axis="y", linestyle="--", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 4 (Bottom-Right): Zero-Shot Difficulty Generalization
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    diff_keys = list(data["difficulty_generalization"].keys())
    short_labels = ["Easy\n(Gap 160, Spd 3.0)", "Standard\n(Gap 140, Spd 3.5)", "Hard\n(Gap 115, Spd 4.5)"]
    xd = np.arange(len(diff_keys))

    for idx, algo in enumerate(algos):
        vals = [data["difficulty_generalization"][dk][algo]["avg_score"] for dk in diff_keys]
        bars = ax4.bar(
            xd + (idx - 1) * width,
            vals,
            width,
            label=algo,
            color=palette[algo],
            edgecolor="black",
        )
        for b in bars:
            h = b.get_height()
            ax4.text(
                b.get_x() + b.get_width() / 2.0,
                h + 1.0,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax4.set_title("4. Zero-Shot Generalization Across Difficulty Tiers", fontsize=12, fontweight="bold")
    ax4.set_xticks(xd)
    ax4.set_xticklabels(short_labels, fontsize=10, fontweight="bold")
    ax4.set_ylabel("Mean Pipes Passed (30 Episodes)", fontsize=11)
    ax4.set_ylim(0, max(85, ax4.get_ylim()[1] * 1.15))
    ax4.legend(loc="upper right", frameon=True)
    ax4.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"Saved ablation figure to: {output_plot}")

    # Print formatted tables
    print("\n" + "=" * 95)
    print(" TABLE 1: MULTI-SEED STABILITY SUMMARY (SEEDS 42, 123, 456 | 30 EPISODES PER SEED)")
    print("=" * 95)
    print(f"{'Algorithm':<10} | {'Seed 42 Avg':<14} | {'Seed 123 Avg':<14} | {'Seed 456 Avg':<14} | {'Across-Seed Mean±Std':<22} | {'Peak Max'}")
    print("-" * 95)
    for a in algos:
        s42 = data["multi_seed"][a]["42"]["eval_avg"]
        s123 = data["multi_seed"][a]["123"]["eval_avg"]
        s456 = data["multi_seed"][a]["456"]["eval_avg"]
        ms = data["multi_seed_summary"][a]
        m_str = f"{ms['mean_across_seeds']:.2f} ± {ms['std_across_seeds']:.2f}"
        print(f"{a:<10} | {s42:<14.2f} | {s123:<14.2f} | {s456:<14.2f} | {m_str:<22} | {ms['max_across_seeds']}")
    print("=" * 95)

    print("\n" + "=" * 95)
    print(" TABLE 2: REWARD SHAPING & OBSERVATION ABLATIONS (30-EPISODE DETERMINISTIC EVAL)")
    print("=" * 95)
    print(f"{'Algorithm':<10} | {'Baseline (5D+Dense)':<21} | {'Sparse Reward Only':<21} | {'Reduced 4D (No vy)':<21}")
    print("-" * 95)
    for a in algos:
        base = data["reward_ablation"][f"{a}_Dense"]
        spar = data["reward_ablation"][f"{a}_Sparse"]
        red = data["observation_ablation"][f"{a}_Reduced4D"]
        b_str = f"{base['eval_avg']:.2f} (max {base['eval_max']})"
        s_str = f"{spar['eval_avg']:.2f} (max {spar['eval_max']})"
        r_str = f"{red['eval_avg']:.2f} (max {red['eval_max']})"
        print(f"{a:<10} | {b_str:<21} | {s_str:<21} | {r_str:<21}")
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(description="Plot Checkpoint 6 & 7 ablation and multi-seed figures")
    parser.add_argument("--input", type=str, default="logs/checkpoint6_experiments.json")
    parser.add_argument("--output", type=str, default="docs/ablation_studies.png")
    args = parser.parse_args()
    generate_ablation_plots(args.input, args.output)


if __name__ == "__main__":
    main()

