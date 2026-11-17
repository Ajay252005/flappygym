"""Interactive Streamlit Web Demo for the Flappy Bird Neuroevolution & Deep RL Benchmark.

Launch with:
    streamlit run app.py
"""

import os
import json
import time
import pickle

# Ensure SDL runs headlessly when invoked inside Streamlit's worker thread on macOS
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import neat
import numpy as np
import pandas as pd
import streamlit as st
from stable_baselines3 import DQN, PPO

import src.env
from src.env.flappy_env import FlappyBirdEnv


@st.cache_resource
def load_cached_agent(agent_name: str):
    """Cache loaded models in memory for instant Streamlit re-runs."""
    if "NEAT" in agent_name:
        cfg = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            "config/neat_config.txt",
        )
        with open("models/neat_best.pkl", "rb") as f:
            winner = pickle.load(f)
        return ("neat", neat.nn.FeedForwardNetwork.create(winner, cfg))
    elif "PPO" in agent_name:
        return ("sb3", PPO.load("models/ppo_best.zip"))
    elif "DQN" in agent_name:
        return ("sb3", DQN.load("models/dqn_best.zip"))
    return ("random", None)


st.set_page_config(
    page_title="flappygym: Flappy Bird Gymnasium Environment (NEAT, DQN, PPO)",
    page_icon="🐦",
    layout="wide",
)

st.title("🐦 flappygym")
st.markdown(
    "A custom Flappy Bird **Gymnasium** environment with **NEAT**, **DQN** and **PPO** agents, "
    "multi-seed evaluation, ablation studies and a Streamlit demo."
)

tab_demo, tab_curves, tab_ablations, tab_findings = st.tabs(
    [
        "🎮 Live Agent Simulator & GIFs",
        "📈 Learning Curves & Benchmark Table",
        "🔬 Checkpoint 6: Ablations & Multi-Seed",
        "📝 Checkpoint 7: Findings & Failure Analysis",
    ]
)

# =============================================================================
# TAB 1: LIVE AGENT SIMULATOR & DEMO GIF GALLERY
# =============================================================================
with tab_demo:
    col_sim, col_ctrl = st.columns([1.2, 1.0])

    with col_ctrl:
        st.subheader("⚙️ Simulation Controls")
        agent_choice = st.selectbox(
            "Select AI Policy",
            ["PPO (Trained - Best Overall)", "NEAT (Trained - Neuroevolution)", "DQN (Trained - Value Based)", "Random Agent (Untrained Baseline)"],
        )
        pipe_gap = st.slider("Pipe Gap Size (px) — Lower is Harder", min_value=100, max_value=180, value=140, step=5)
        pipe_speed = st.slider("Pipe Scroll Speed (px/frame)", min_value=2.5, max_value=5.5, value=3.5, step=0.5)
        max_pipes = st.slider("Stop Episode After N Pipes", min_value=10, max_value=100, value=30, step=10)
        sim_seed = st.number_input("Episode Random Seed", min_value=1, max_value=99999, value=1003, step=1)
        run_btn = st.button("▶️ Run Live Episode in Browser", type="primary", width="stretch")

        st.markdown("---")
        st.markdown("#### 🎞️ Pre-Recorded Demo GIFs (Untrained vs. Trained)")
        gif_cols = st.columns(2)
        with gif_cols[0]:
            if os.path.exists("docs/random_agent_demo.gif"):
                st.image("docs/random_agent_demo.gif", caption="Untrained Random Agent (0 pipes, ~38 frames)", width="stretch")
            if os.path.exists("docs/dqn_trained_demo.gif"):
                st.image("docs/dqn_trained_demo.gif", caption="Trained DQN Agent (Avg 15.8, Max 47 pipes)", width="stretch")
        with gif_cols[1]:
            if os.path.exists("docs/neat_trained_demo.gif"):
                st.image("docs/neat_trained_demo.gif", caption="Trained NEAT Agent (Avg 31.9, Max 100 pipes)", width="stretch")
            if os.path.exists("docs/ppo_trained_demo.gif"):
                st.image("docs/ppo_trained_demo.gif", caption="Trained PPO Agent (Avg 59.6, Max 100 pipes)", width="stretch")

    with col_sim:
        st.subheader("🖥️ Live Environment Viewport")
        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()

        if run_btn:
            env = FlappyBirdEnv(
                render_mode="rgb_array",
                normalize_obs=True,
                pipe_gap_size=pipe_gap,
                pipe_speed=pipe_speed,
            )
            obs, info = env.reset(seed=int(sim_seed))

            agent_kind, loaded_model = load_cached_agent(agent_choice)
            net = loaded_model if agent_kind == "neat" else None
            sb3_model = loaded_model if agent_kind == "sb3" else None

            terminated = False
            truncated = False
            step_idx = 0

            while not (terminated or truncated):
                if net is not None:
                    out = net.activate(obs)
                    action = 1 if out[0] > 0.0 else 0
                elif sb3_model is not None:
                    act, _ = sb3_model.predict(obs, deterministic=True)
                    action = int(act)
                else:
                    action = int(env.action_space.sample())

                obs, reward, terminated, truncated, info = env.step(action)
                step_idx += 1

                if step_idx % 2 == 0 or terminated or info["score"] >= max_pipes:
                    frame = env.render()
                    if frame is not None:
                        frame_placeholder.image(frame, caption=f"{agent_choice} — Step {step_idx}", width=400)
                    metrics_placeholder.markdown(
                        f"**Pipes Cleared:** `{info['score']}` &nbsp;|&nbsp; "
                        f"**Frames Survived:** `{info['frames_survived']}` &nbsp;|&nbsp; "
                        f"**Action:** `{'FLAP ⬆️' if action == 1 else 'GLIDE ⬇️'}`"
                    )
                    time.sleep(0.02)

                if info["score"] >= max_pipes:
                    st.success(f"🎉 Target reached! Agent cleared {info['score']} pipes without crashing!")
                    break

            if terminated:
                st.info(f"Episode finished: Agent cleared **{info['score']} pipes** ({info['frames_survived']} frames).")
            env.close()
        else:
            if os.path.exists("docs/untrained_vs_trained_comparison.gif"):
                frame_placeholder.image(
                    "docs/untrained_vs_trained_comparison.gif",
                    caption="4-Agent Synchronized Comparison (Random vs. NEAT vs. DQN vs. PPO) — Click 'Run Live Episode in Browser' above!",
                    width="stretch",
                )
            elif os.path.exists("docs/ppo_trained_demo.gif"):
                frame_placeholder.image("docs/ppo_trained_demo.gif", caption="Click 'Run Live Episode in Browser' to run custom simulations!", width=400)

# =============================================================================
# TAB 2: LEARNING CURVES & BENCHMARK TABLE
# =============================================================================
with tab_curves:
    st.subheader("📊 Comparative Benchmark: Environment Steps vs. Score (Seed 42 Main Run)")
    if os.path.exists("docs/learning_curve_comparison.png"):
        st.image("docs/learning_curve_comparison.png", width="stretch")

    st.markdown("### 30-Episode Deterministic Evaluation Summary")
    benchmark_df = pd.DataFrame(
        [
            {
                "Algorithm": "PPO (8 Parallel Envs)",
                "Paradigm": "On-Policy Actor-Critic (Clipped Policy Gradient)",
                "Total Env Steps": "1,015,808",
                "Eval Mean ± Std (30 Eps)": "59.57 ± 38.39",
                "Eval Max Score": "100 (Cap)",
                "Train Peak Score": 184,
                "Wall-Clock Time": "~3.8 min",
            },
            {
                "Algorithm": "NEAT (Pop=100, 60 Gens)",
                "Paradigm": "Neuroevolution of Augmenting Topologies",
                "Total Env Steps": "1,533,253",
                "Eval Mean ± Std (30 Eps)": "31.87 ± 28.93",
                "Eval Max Score": "100 (Cap)",
                "Train Peak Score": 70,
                "Wall-Clock Time": "~20.5 sec",
            },
            {
                "Algorithm": "DQN (Replay=100k)",
                "Paradigm": "Off-Policy Value-Based TD Learning",
                "Total Env Steps": "1,000,000",
                "Eval Mean ± Std (30 Eps)": "15.80 ± 13.12",
                "Eval Max Score": "47",
                "Train Peak Score": 16,
                "Wall-Clock Time": "~5.0 min",
            },
            {
                "Algorithm": "Random Baseline",
                "Paradigm": "Uniform Random Actions",
                "Total Env Steps": "5,000",
                "Eval Mean ± Std (30 Eps)": "0.00 ± 0.00",
                "Eval Max Score": "0",
                "Train Peak Score": 0,
                "Wall-Clock Time": "0 sec",
            },
        ]
    )
    st.table(benchmark_df)

# =============================================================================
# TAB 3: CHECKPOINT 6 ABLATIONS & MULTI-SEED STABILITY
# =============================================================================
with tab_ablations:
    st.subheader("🔬 Controlled Ablations & Multi-Seed Robustness")
    if os.path.exists("docs/ablation_studies.png"):
        st.image("docs/ablation_studies.png", width="stretch")

    if os.path.exists("logs/checkpoint6_experiments.json"):
        with open("logs/checkpoint6_experiments.json") as f:
            exp_data = json.load(f)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 1. Multi-Seed Stability (Seeds 42, 123, 456)")
            ms_rows = []
            for algo in ["PPO", "NEAT", "DQN"]:
                s42 = exp_data["multi_seed"][algo]["42"]["eval_avg"]
                s123 = exp_data["multi_seed"][algo]["123"]["eval_avg"]
                s456 = exp_data["multi_seed"][algo]["456"]["eval_avg"]
                summ = exp_data["multi_seed_summary"][algo]
                ms_rows.append(
                    {
                        "Algorithm": algo,
                        "Seed 42": f"{s42:.2f}",
                        "Seed 123": f"{s123:.2f}",
                        "Seed 456": f"{s456:.2f}",
                        "Across-Seed Mean ± Std": f"{summ['mean_across_seeds']:.2f} ± {summ['std_across_seeds']:.2f}",
                        "Peak Max": summ["max_across_seeds"],
                    }
                )
            st.table(pd.DataFrame(ms_rows))

        with col_b:
            st.markdown("#### 2. Reward & Observation Ablations")
            ab_rows = []
            for algo in ["PPO", "NEAT", "DQN"]:
                d_val = exp_data["reward_ablation"][f"{algo}_Dense"]
                s_val = exp_data["reward_ablation"][f"{algo}_Sparse"]
                r_val = exp_data["observation_ablation"][f"{algo}_Reduced4D"]
                ab_rows.append(
                    {
                        "Algorithm": algo,
                        "Full 5D + Dense (Baseline)": f"{d_val['eval_avg']:.2f} (max {d_val['eval_max']})",
                        "Sparse Reward Only": f"{s_val['eval_avg']:.2f} (max {s_val['eval_max']})",
                        "Reduced 4D (No Velocity)": f"{r_val['eval_avg']:.2f} (max {r_val['eval_max']})",
                    }
                )
            st.table(pd.DataFrame(ab_rows))

# =============================================================================
# TAB 4: CHECKPOINT 7 RESEARCH WRITE-UP
# =============================================================================
with tab_findings:
    if os.path.exists("docs/findings_report.md"):
        with open("docs/findings_report.md") as f:
            md_lines = [line for line in f.read().splitlines() if not line.strip().startswith("![")]
            st.markdown("\n".join(md_lines))
    else:
        st.info("Findings report (`docs/findings_report.md`) will appear here once generated.")

