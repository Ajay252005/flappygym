"""Flappy Bird game module."""

from src.game.constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    GROUND_Y,
    PIPE_GAP_SIZE,
    PIPE_SPEED,
    FPS,
)
from src.game.bird import Bird
from src.game.pipe import PipePair
from src.game.ground import Ground
from src.game.flappy_game import FlappyBirdGame, GameState
from src.game.renderer import FlappyRenderer

__all__ = [
    "SCREEN_WIDTH",
    "SCREEN_HEIGHT",
    "GROUND_Y",
    "PIPE_GAP_SIZE",
    "PIPE_SPEED",
    "FPS",
    "Bird",
    "PipePair",
    "Ground",
    "FlappyBirdGame",
    "GameState",
    "FlappyRenderer",
]

