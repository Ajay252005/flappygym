"""Bird physics and animation representation."""

import pygame
from src.game.constants import (
    BIRD_X,
    BIRD_START_Y,
    BIRD_WIDTH,
    BIRD_HEIGHT,
    BIRD_HITBOX_PADDING,
    GRAVITY,
    FLAP_STRENGTH,
    MAX_DROP_SPEED,
    MAX_UP_SPEED,
)


class Bird:
    """Represents the player-controlled or RL-controlled bird."""

    def __init__(
        self,
        x: float = BIRD_X,
        y: float = BIRD_START_Y,
        gravity: float = GRAVITY,
        flap_strength: float = FLAP_STRENGTH,
    ):
        self.start_x = float(x)
        self.start_y = float(y)
        self.width = BIRD_WIDTH
        self.height = BIRD_HEIGHT
        self.hitbox_padding = BIRD_HITBOX_PADDING

        self.gravity = gravity
        self.flap_strength = flap_strength
        self.max_drop_speed = MAX_DROP_SPEED
        self.max_up_speed = MAX_UP_SPEED

        self.x = self.start_x
        self.y = self.start_y
        self.velocity = 0.0
        self.rotation = 0.0  # In degrees: positive is nose up, negative is nose down
        self.flap_anim_timer = 0
        self.wing_state = 0  # 0: mid, 1: up, 2: down

    def flap(self) -> None:
        """Trigger upward flap impulse."""
        self.velocity = self.flap_strength
        self.rotation = 25.0
        self.wing_state = 1

    def update(self) -> None:
        """Update bird position, velocity, and rotation angle by one tick."""
        # Apply gravity
        self.velocity += self.gravity
        if self.velocity > self.max_drop_speed:
            self.velocity = self.max_drop_speed
        elif self.velocity < self.max_up_speed:
            self.velocity = self.max_up_speed

        self.y += self.velocity

        # Rotation mechanics:
        # If moving up or near peak, tilt up. When falling, rotate smoothly towards -90 degrees.
        if self.velocity < 2.0:
            self.rotation = 25.0
        else:
            self.rotation -= 3.5
            if self.rotation < -90.0:
                self.rotation = -90.0

        # Animate wings while not in nosedive
        if self.rotation > -60.0:
            self.flap_anim_timer += 1
            if self.flap_anim_timer % 5 == 0:
                self.wing_state = (self.wing_state + 1) % 3
        else:
            self.wing_state = 0

    def get_hitbox(self) -> pygame.Rect:
        """Returns the collision rectangle for the bird with minor padding for fairness."""
        pad = self.hitbox_padding
        return pygame.Rect(
            int(self.x + pad),
            int(self.y + pad),
            int(self.width - 2 * pad),
            int(self.height - 2 * pad),
        )

    def reset(self, y: float | None = None) -> None:
        """Reset bird to initial state."""
        self.x = self.start_x
        self.y = float(y) if y is not None else self.start_y
        self.velocity = 0.0
        self.rotation = 0.0
        self.flap_anim_timer = 0
        self.wing_state = 0

