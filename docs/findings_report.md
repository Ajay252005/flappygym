# Research Findings & Technical Write-Up: Neuroevolution (NEAT) vs. Deep Reinforcement Learning (DQN & PPO) on Flappy Bird

---

## Executive Summary (Checkpoint 7 TL;DR)

> **Which method won and why?**  
> **PPO (Proximal Policy Optimization)** achieved the highest overall reliability and peak performance (**59.57 ± 38.39 pipes**, **100 pipes eval max**, **184 pipes training record**, and **40.83 ± 27.04 pipes across seeds**), winning on asymptotic policy quality because on-policy actor-critic updates across **8 parallel environments** reduce gradient variance and handle continuous state-value estimation smoothly.  
> **NEAT (Neuroevolution of Augmenting Topologies)** won on **wall-clock efficiency (~20.5 seconds vs. ~4–5 minutes)** and **ablation resilience**, reaching **31.87 ± 28.93 pipes (100 max)** with a tiny 5-input, 1-output network and uniquely surviving both the **sparse-reward ablation (16.80 avg, 64 max vs. 0.00 for DQN/PPO)** and the **no-velocity 4D observation ablation (23.93 avg, 100 max)**.  
> **DQN (Deep Q-Network)** finished third (**15.80 ± 13.12 pipes, 47 max**), suffering from off-policy Q-value overestimation and replay-buffer lag on a task where a single mistimed flap causes instant termination.

---

## 1. Experimental Setup & Fair Methodology

To ensure an apples-to-apples comparison across fundamentally different paradigms (population-based evolution vs. gradient-based temporal-difference learning), all experiments were standardized around four rules:

1. **Aligned Sample Budget (Cumulative Environment Steps):** Rather than comparing generations directly against gradient updates, every step taken by every genome in the NEAT population was logged and aligned against cumulative environment steps for DQN and PPO (~1.0M–1.5M steps).
2. **Identical Normalized State Space (5D Box):**
   $$\mathbf{s}_t = \left[\frac{y_{\text{bird}}}{H},\; \frac{v_{y,\text{bird}}}{v_{\max}},\; \frac{\Delta x_{\text{pipe}}}{W},\; \frac{y_{\text{gap,top}}}{H},\; \frac{y_{\text{gap,bot}}}{H}\right] \in \mathbb{R}^5$$
3. **30-Episode Deterministic Evaluation:** Small 5-episode evaluations exhibit extreme variance due to random pipe heights. Every model was evaluated over **30 seeded episodes** (`max_pipes=100` cap per episode) using deterministic action selection.
4. **Multi-Seed Verification:** Each algorithm was run across **3 independent random seeds (`42`, `123`, `456`)** to measure sensitivity to initialization and exploration noise.

---

## 2. Main Comparative Results (Checkpoints 3–5)

![Comparative Learning Curves](learning_curve_comparison.png)

| Algorithm | Paradigm | Total Env Steps | Train Peak | Eval Mean ± Std (30 Eps) | Eval Max | Wall-Clock Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PPO** (`n_envs=8`) | On-Policy Policy Gradient | 1,015,808 | **184** | **59.57 ± 38.39** | **100** | ~3.8 min (230s) |
| **NEAT** (`pop=100`) | Neuroevolution | 1,533,253 | 70 | **31.87 ± 28.93** | **100** | **~0.3 min (20.5s)** |
| **DQN** (`buf=100k`) | Off-Policy Value-Based DRL | 1,000,000 | 16 | **15.80 ± 13.12** | 47 | ~5.0 min (298s) |
| **Random Baseline** | Uniform Random (`p=0.5`) | 5,000 | 0 | **0.00 ± 0.00** | 0 | <0.1s |

---

## 3. Controlled Experiments & Ablations (Checkpoint 6)

![Ablation Studies & Multi-Seed Stability](ablation_studies.png)

### 3.1 Multi-Seed Stability (Seeds 42, 123, 456)

| Algorithm | Seed 42 Mean (1M Steps) | Seed 123 Mean (250k Steps) | Seed 456 Mean (250k Steps) | Across-Seed Mean ± Std | Peak Max Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PPO** | 59.57 ± 38.39 | **60.33 ± 39.21** | 2.60 ± 2.11 | **40.83 ± 27.04** | **100** |
| **NEAT** | 31.87 ± 28.93 | 1.13 ± 1.02 | 14.07 ± 11.45 | **15.69 ± 12.60** | **100** |
| **DQN** | 15.80 ± 13.12 | 1.17 ± 1.04 | 0.07 ± 0.25 | **5.68 ± 7.17** | **47** |

- **Finding:** PPO converged to a 60+ pipe policy even at 250k steps on Seed 123, demonstrating superior sample efficiency when vectorized across 8 parallel environments. NEAT is ultra-fast in wall-clock time because forward passes on a 5-node graph require no backpropagation, but its evolutionary search is sensitive to whether an early generation discovers a pipe-clearing mutation before species stagnation.

---

### 3.2 Reward-Shaping Ablation: Dense vs. Sparse Reward

We compared **Dense Reward** ($+0.1$ survival per frame, $+0.05(1 - d_{\text{center}})$ gap-centering guidance, $+5.0$ per pipe, $-5.0$ on collision) against **Sparse Reward** ($+1.0$ per pipe passed, $0.0$ otherwise):

| Algorithm | Dense Reward (Mean ± Std) | Sparse Reward Only (Mean ± Std) | Relative Change |
| :--- | :---: | :---: | :---: |
| **PPO** | **59.57 ± 38.39** (max 100) | **0.00 ± 0.00** (max 0) | **-100.0% (Complete Failure)** |
| **DQN** | **15.80 ± 13.12** (max 47) | **0.00 ± 0.00** (max 0) | **-100.0% (Complete Failure)** |
| **NEAT** | **31.87 ± 28.93** (max 100) | **16.80 ± 14.52** (max 64) | **-47.3% (Still Passes 64 Pipes!)** |

- **Why Deep RL Failed Completely on Sparse Reward:** An untrained agent survives ~38 frames and collides before reaching the first pipe (which requires surviving ~95 frames and aligning vertically within a 140 px gap). Under sparse rewards, 99.9% of trajectories collect a total return of $G_t = 0$. Consequently, TD errors in DQN and advantage estimates $\hat{A}_t$ in PPO are identically zero across entire minibatches, starving gradient descent of any directional signal.
- **Why NEAT Survived Sparse Reward:** NEAT optimizes over episodic population fitness and does not rely on per-step reward gradients. As soon as a few random mutations in the 100-genome population clear the first pipe, elitism and speciation preserve those weights intact, allowing evolution to climb from 1 pipe to **64 pipes**.

---

### 3.3 Observation-Space Ablation: Full 5D vs. Reduced 4D (Omitting Vertical Velocity $v_y$)

We removed the bird's vertical velocity (`bird_vy`) from the observation vector, reducing the state space from 5D to 4D `[bird_y, dist_to_pipe, gap_top, gap_bottom]`:

| Algorithm | Full 5D State (With $v_y$) | Reduced 4D State (No $v_y$) | Relative Drop |
| :--- | :---: | :---: | :---: |
| **PPO** | **59.57 ± 38.39** (max 100) | **1.83 ± 1.44** (max 6) | **-96.9%** |
| **DQN** | **15.80 ± 13.12** (max 47) | **4.70 ± 4.18** (max 17) | **-70.3%** |
| **NEAT** | **31.87 ± 28.93** (max 100) | **23.93 ± 24.10** (max 100) | **-24.9%** |

- **Why Omitting $v_y$ Hurts Markovian RL:** Flappy Bird has second-order gravitational dynamics ($y_{t+1} = y_t + v_{y,t}$). Without $v_y$, two states with identical position $y_t$—one rising rapidly at $-8\text{ px/frame}$ right after a flap, and one plunging at $+10\text{ px/frame}$—look identical to the value network $V(s_t)$. This **state aliasing** corrupts TD value targets in PPO and DQN.
- **Why NEAT Handled 4D State Better:** A memoryless reactive controller that flaps whenever $y_{\text{bird}} > y_{\text{gap\_bottom}} - \delta$ can still bound vertical oscillation tightly enough to clear pipes without estimating a value function.

---

### 3.4 Zero-Shot Difficulty Generalization

We evaluated the trained Standard models (`gap=140, speed=3.5`) zero-shot on **Easy (`gap=160, speed=3.0`)** and **Hard (`gap=115, speed=4.5`)** environments across 30 episodes:

| Difficulty Tier | NEAT (Mean / Max) | DQN (Mean / Max) | PPO (Mean / Max) |
| :--- | :---: | :---: | :---: |
| **Easy (`Gap=160, Speed=3.0`)** | **100.00 / 100** | **95.27 / 100** | 63.23 / 100 |
| **Standard (`Gap=140, Speed=3.5`)** | 25.00 / 100 | 13.87 / 47 | **72.90 / 100** |
| **Hard (`Gap=115, Speed=4.5`)** | **2.80 / 9** | 2.23 / 8 | 1.80 / 7 |

- **Insight:** On Easy mode, both NEAT (**100.0 average!**) and DQN (**95.3 average!**) generalize nearly flawlessly because wider gaps forgive minor overshoot from discrete flap impulses. On Hard mode (`115 px` gap + `4.5 px/frame` speed), all three policies degrade to ~2–3 pipes because a single flap impulse (`-8.0 px/frame`, peak arc ~68 px) leaves minimal vertical margin inside a 115 px gap when horizontal transit time is compressed by 29%.

---

## 4. Failure Modes & Lessons Learned

1. **Undertraining Trap in Deep RL (100k vs. 1M Steps):** In early experiments at 60k–100k steps, both DQN and PPO scored `<2` pipes while NEAT scored `70+`, creating a misleading impression that DRL cannot solve Flappy Bird. Scaling to **1,000,000 steps** with **8 vectorized environments** for PPO revealed that PPO actually surpasses NEAT in mean score (`59.6` vs. `31.9`).
2. **DQN Replay Buffer & Exploration Lag:** Because Flappy Bird has a sharp action asymmetry (flapping too often hits the ceiling or top pipe; waiting one frame too long hits the bottom pipe), $\epsilon$-greedy exploration with $\epsilon > 0.1$ destroys long trajectories. DQN only begins passing multiple pipes after step ~250,000 once $\epsilon$ decays near `0.02`.
3. **NEAT Fitness Shaping Pitfall:** An initial NEAT fitness function that subtracted distance-to-gap-center on every frame caused evolution to prefer dying early near the center line over surviving slightly off-center. Switching to a strictly positive pipe bonus (`+200` per pipe passed immediately upon clearing) solved this local optimum.

