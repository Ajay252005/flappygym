"""Pipe pair management and collision geometry."""

import random
import pygame
from src.game.constants import (
    PIPE_WIDTH,
    PIPE_GAP_SIZE,
    PIPE_MIN_TOP_HEIGHT,
    PIPE_SPEED,
    GROUND_Y,
)


class PipePair:
    """Represents a pair of pipes (top and bottom) with a vertical gap between them."""

    def __init__(
        self,
        x: float,
        gap_size: int = PIPE_GAP_SIZE,
        speed: float = PIPE_SPEED,
        gap_top: float | None = None,
        rng: random.Random | None = None,
    ):
        self.x = float(x)
        self.width = PIPE_WIDTH
        self.gap_size = gap_size
        self.speed = speed
        self.passed = False

        min_top = PIPE_MIN_TOP_HEIGHT
        max_top = GROUND_Y - PIPE_MIN_TOP_HEIGHT - self.gap_size

        if gap_top is not None:
            self.gap_top = float(gap_top)
        else:
            picker = rng if rng is not None else random
            self.gap_top = float(picker.randint(min_top, max(min_top, max_top)))

        self.gap_bottom = self.gap_top + self.gap_size

    def update(self) -> None:
        """Move pipe pair to the left by its speed."""
        self.x -= self.speed

    def get_top_rect(self) -> pygame.Rect:
        """Rectangle defining the upper pipe obstacle."""
        return pygame.Rect(int(self.x), 0, self.width, int(self.gap_top))

    def get_bottom_rect(self) -> pygame.Rect:
        """Rectangle defining the lower pipe obstacle."""
        return pygame.Rect(
            int(self.x),
            int(self.gap_bottom),
            self.width,
            max(0, int(GROUND_Y - self.gap_bottom)),
        )

    def collides_with(self, bird_rect: pygame.Rect) -> bool:
        """Check whether the given bird rectangle intersects either pipe."""
        return (
            bird_rect.colliderect(self.get_top_rect())
            or bird_rect.colliderect(self.get_bottom_rect())
        )

    def is_offscreen(self) -> bool:
        """Check if pipe pair has moved completely off the left of the screen."""
        return self.x + self.width < 0

