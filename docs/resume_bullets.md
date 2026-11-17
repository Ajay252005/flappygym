# Resume Bullets: Flappy Bird Neuroevolution vs. Deep RL Benchmark

Below are ready-to-use, measurable resume bullet points tailored for **Machine Learning Engineer**, **Reinforcement Learning Researcher**, and **Software Engineer (AI/ML)** roles.

---

## Option A — Concise 3-Bullet Format (Recommended for 1-Page Tech Resume)

**Comparing Neuroevolution (NEAT) & Deep Reinforcement Learning (DQN, PPO) on Custom Flappy Bird** | *Python, PyTorch, Gymnasium, Stable-Baselines3, NEAT-Python, Pygame, Streamlit*
- **Engineered a Gymnasium-compliant RL environment** (`Discrete(2)`, normalized 5D state space) on top of a custom Pygame physics engine running at **52,000+ headless steps/sec**, validated via **31 automated `pytest` unit/integration tests**.
- **Benchmarked NEAT, DQN, and PPO across 1M–1.5M environment steps and 3 random seeds (90 evaluation episodes per algorithm)**, achieving **59.6 ± 38.4 pipes (100 max)** with 8-env vectorized PPO, **31.9 ± 28.9 pipes (100 max)** in **20.5s wall-clock training** with NEAT, and **15.8 ± 13.1 pipes (47 max)** with DQN.
- **Conducted controlled reward-shaping, state-space, and zero-shot generalization ablations**, demonstrating that sparse rewards collapse TD/policy-gradient methods (**-100% score**) while NEAT retains **64-pipe peak performance**, and that omitting vertical velocity degrades PPO by **96.9%** due to non-Markovian state aliasing; deployed an interactive **Streamlit live simulation dashboard**.

---

## Option B — High-Impact 4-Bullet Format (For ML / RL Specialist Roles)

**Neuroevolution vs. Deep RL Benchmark on Custom Flappy Bird Environment** | *Python, PyTorch, Gymnasium, Stable-Baselines3, NEAT, TensorBoard, Streamlit*
- **Custom High-Throughput Simulation & Gym Wrapper:** Built a deterministic Flappy Bird engine and registered 3 `Gymnasium` environments (`FlappyBird-v0`, `-Reduced-v0`, `-Sparse-v0`) achieving **52,400+ FPS** in headless mode with full `check_env` compliance and RGB array rendering.
- **Fair Sample-Aligned RL & Neuroevolution Training:** Trained and evaluated **PPO** (`n_envs=8`, GAE $\lambda=0.95$), **DQN** (`100k` replay buffer, target network updates), and **NEAT** (`pop=100`, topology/weight speciation) aligned on **1M+ cumulative environment steps**, reaching **184 training / 100 eval max pipes** with PPO and **35x faster wall-clock convergence (20.5s vs. 4–5 min)** with NEAT.
- **Ablation & Robustness Analysis Across 3 Seeds:** Executed 12+ training runs across multiple seeds, proving that dense gap-centering reward shaping is essential for Deep RL (**59.6 vs. 0.0 pipes on sparse reward**) whereas population-based neuroevolution solves sparse rewards (**16.8 avg / 64 max pipes**) and tolerates partial observability (**23.9 avg pipes without velocity input vs. 1.8 for PPO**).
- **Interactive Web Demo & Reproducible Suite:** Built a live **Streamlit** web application with real-time policy execution, difficulty stress-testing sliders (`Easy` **100.0 avg** to `Hard` **2.8 avg**), synchronized 4-agent comparison GIFs, and a **31-test** continuous verification suite.

---

## Option C — Ultra-Short 2-Bullet Version (When Space is Tight)

- **Built a custom 52k-FPS Gymnasium Flappy Bird benchmark** comparing **PPO**, **DQN**, and **NEAT** across **1M+ environment steps and 3 seeds**, achieving **59.6 avg / 100 max pipes** with 8-env vectorized PPO and **20.5s training convergence** with NEAT (**31.9 avg / 100 max pipes**).
- **Performed reward-shaping (dense vs. sparse) and Markovian state-space (5D vs. 4D no-velocity) ablations** across **90+ deterministic evaluation rollouts**, uncovering a **96.9% PPO drop** under velocity state aliasing while NEAT maintained **64-pipe sparse-reward** performance; shipped a live **Streamlit** demo and **31-test** `pytest` suite.

