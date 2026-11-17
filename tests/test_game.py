"""Unit tests verifying Flappy Bird game mechanics, physics, scoring, collisions, and state."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from src.game.constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    GROUND_Y,
    BIRD_START_Y,
    FLAP_STRENGTH,
    MAX_DROP_SPEED,
)
from src.game.bird import Bird
from src.game.pipe import PipePair
from src.game.ground import Ground
from src.game.flappy_game import FlappyBirdGame, GameState


class TestBirdPhysics:
    def test_bird_initialization(self):
        bird = Bird()
        assert bird.x == 80
        assert bird.y == BIRD_START_Y
        assert bird.velocity == 0.0
        assert bird.rotation == 0.0

    def test_bird_gravity_fall(self):
        bird = Bird()
        initial_y = bird.y
        bird.update()
        assert bird.velocity > 0  # Falling downward
        assert bird.y > initial_y

    def test_bird_flap(self):
        bird = Bird()
        # Fall a bit
        for _ in range(5):
            bird.update()
        assert bird.velocity > 0

        # Flap
        bird.flap()
        assert bird.velocity == FLAP_STRENGTH
        assert bird.rotation == 25.0

    def test_bird_terminal_velocity(self):
        bird = Bird()
        for _ in range(100):
            bird.update()
        assert bird.velocity == MAX_DROP_SPEED


class TestPipesAndGround:
    def test_pipe_pair_initialization(self):
        pipe = PipePair(x=300, gap_size=140, speed=3.5)
        assert pipe.x == 300
        assert pipe.gap_bottom - pipe.gap_top == 140
        assert not pipe.passed

    def test_pipe_movement(self):
        pipe = PipePair(x=300, speed=4.0)
        pipe.update()
        assert pipe.x == 296.0

    def test_ground_collision(self):
        ground = Ground()
        bird = Bird()
        bird.y = GROUND_Y + 50
        assert ground.collides_with(bird.get_hitbox())

    def test_ground_no_collision_when_flying(self):
        ground = Ground()
        bird = Bird()
        bird.y = 200
        assert not ground.collides_with(bird.get_hitbox())


class TestGameEngine:
    def test_game_reset(self):
        game = FlappyBirdGame(seed=42)
        game.reset()
        assert game.score == 0
        assert game.frames_survived == 0
        assert game.game_state == GameState.PLAYING
        assert not game.collision_detected
        assert len(game.pipes) >= 2
        # Pipes should be ahead of bird
        assert game.pipes[0].x > game.bird.x

    def test_observation_dimensions(self):
        game = FlappyBirdGame(seed=42)
        game.reset()

        obs_full = game.get_observation(reduced=False)
        assert isinstance(obs_full, np.ndarray)
        assert obs_full.shape == (5,)
        # [bird_y, bird_vy, dist_to_pipe, gap_top, gap_bottom]
        assert obs_full[0] == pytest.approx(game.bird.y, rel=1e-3)
        assert obs_full[1] == pytest.approx(game.bird.velocity, rel=1e-3)

        obs_reduced = game.get_observation(reduced=True)
        assert obs_reduced.shape == (4,)
        # [bird_y, dist_to_pipe, gap_top, gap_bottom] without velocity
        assert obs_reduced[0] == pytest.approx(game.bird.y, rel=1e-3)

    def test_pipe_scoring(self):
        game = FlappyBirdGame(seed=42)
        game.reset()

        # Place a pipe right behind the bird
        game.pipes[0].x = game.bird.x + 2
        assert game.score == 0

        # Step to advance pipe past bird
        game.pipes[0].x = game.bird.x - game.pipes[0].width - 1
        # Set bird safely in middle
        game.bird.y = 200
        game.bird.velocity = 0

        # Run step with flap to keep afloat
        is_over, passed = game.step(action=0)
        assert passed == 1
        assert game.score == 1
        assert game.pipes[0].passed is True

    def test_ground_collision_triggers_game_over(self):
        game = FlappyBirdGame(seed=42)
        game.reset()
        # Drop bird to ground
        game.bird.y = GROUND_Y + 10
        is_over, _ = game.step(action=0)
        assert is_over is True
        assert game.game_state == GameState.GAME_OVER
        assert game.collision_detected is True

    def test_ceiling_collision_triggers_game_over(self):
        game = FlappyBirdGame(seed=42)
        game.reset()
        # Fly bird above top of screen
        game.bird.y = -5
        is_over, _ = game.step(action=0)
        assert is_over is True
        assert game.game_state == GameState.GAME_OVER

    def test_pipe_collision_triggers_game_over(self):
        game = FlappyBirdGame(seed=42)
        game.reset()
        # Position pipe directly overlapping bird at upper pipe
        pipe = game.pipes[0]
        pipe.x = game.bird.x
        game.bird.y = pipe.gap_top - 20  # Inside top pipe

        is_over, _ = game.step(action=0)
        assert is_over is True
        assert game.game_state == GameState.GAME_OVER

    def test_rgb_array_rendering(self):
        game = FlappyBirdGame(seed=42)
        game.reset()
        frame = game.render(mode="rgb_array")
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        assert frame.dtype == np.uint8
        # Ensure frame is not all black
        assert frame.mean() > 50

    def test_seed_reproducibility(self):
        game1 = FlappyBirdGame(seed=12345)
        game1.reset()
        pipes1_tops = [p.gap_top for p in game1.pipes]

        game2 = FlappyBirdGame(seed=12345)
        game2.reset()
        pipes2_tops = [p.gap_top for p in game2.pipes]

        assert pipes1_tops == pipes2_tops

