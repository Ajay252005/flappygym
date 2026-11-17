"""Scrolling ground representation."""

import pygame
from src.game.constants import (
    SCREEN_WIDTH,
    GROUND_HEIGHT,
    GROUND_Y,
    PIPE_SPEED,
)


class Ground:
    """Manages the scrolling ground and checks ground collision."""

    def __init__(self, speed: float = PIPE_SPEED):
        self.y = GROUND_Y
        self.height = GROUND_HEIGHT
        self.speed = speed
        self.scroll_x = 0.0
        self.stripe_width = 24

    def update(self) -> None:
        """Scroll ground leftward seamlessly."""
        self.scroll_x = (self.scroll_x - self.speed) % self.stripe_width

    def collides_with(self, bird_rect: pygame.Rect) -> bool:
        """Returns True if bird hits the ground."""
        return bird_rect.bottom >= self.y

    def reset(self) -> None:
        """Reset scroll offset."""
        self.scroll_x = 0.0

