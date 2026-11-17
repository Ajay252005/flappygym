"""Tests for Checkpoint 6 & 7 experiments, ablation artifacts, and Streamlit demo."""

import os
import sys
import json
import py_compile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestCheckpoints6And7:
    def test_experiment_scripts_compile(self):
        py_compile.compile("src/experiments/compare_learning_curves.py", doraise=True)
        py_compile.compile("src/experiments/run_ablations_and_seeds.py", doraise=True)
        py_compile.compile("src/experiments/plot_ablations.py", doraise=True)
        py_compile.compile("app.py", doraise=True)

    def test_checkpoint6_json_structure(self):
        path = "logs/checkpoint6_experiments.json"
        assert os.path.exists(path), f"{path} missing"
        with open(path, "r") as f:
            data = json.load(f)

        assert "multi_seed" in data
        assert "multi_seed_summary" in data
        assert "reward_ablation" in data
        assert "observation_ablation" in data
        assert "difficulty_generalization" in data

        for algo in ["NEAT", "DQN", "PPO"]:
            assert algo in data["multi_seed"]
            for seed in ["42", "123", "456"]:
                assert seed in data["multi_seed"][algo]
                assert "eval_avg" in data["multi_seed"][algo][seed]

    def test_visual_deliverables_exist(self):
        expected_files = [
            "docs/learning_curve_comparison.png",
            "docs/ablation_studies.png",
            "docs/untrained_vs_trained_comparison.gif",
            "docs/random_agent_demo.gif",
            "docs/neat_trained_demo.gif",
            "docs/dqn_trained_demo.gif",
            "docs/ppo_trained_demo.gif",
            "docs/findings_report.md",
            "docs/resume_bullets.md",
        ]
        for fpath in expected_files:
            assert os.path.exists(fpath), f"Deliverable missing: {fpath}"
            assert os.path.getsize(fpath) > 100, f"Deliverable empty: {fpath}"

    def test_multithreaded_rgb_array_render(self):
        """Ensure offscreen rendering works inside background threads (as used by Streamlit on macOS)."""
        import threading
        from src.env.flappy_env import FlappyBirdEnv

        errors = []

        def worker():
            try:
                env = FlappyBirdEnv(render_mode="rgb_array", normalize_obs=True)
                env.reset(seed=42)
                env.step(1)
                frame = env.render()
                assert frame is not None and frame.shape == (600, 400, 3)
                env.close()
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert not errors, f"Thread render error: {errors}"

    def test_streamlit_app_headless_execution(self):
        """Run Streamlit AppTest on app.py including live simulation button click."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        assert not at.exception, f"Initial render exception: {at.exception}"

        at.selectbox[0].select("Random Agent (Untrained Baseline)").run()
        at.button[0].click().run()
        assert not at.exception, f"Live episode button exception: {at.exception}"



