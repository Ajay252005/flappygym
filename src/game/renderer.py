"""Pygame-based procedural rendering engine for Flappy Bird."""

import math
import numpy as np
import pygame
from src.game.constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    GROUND_Y,
    GROUND_HEIGHT,
    COLOR_SKY,
    COLOR_GROUND_BASE,
    COLOR_GROUND_STRIPE,
    COLOR_GROUND_GRASS,
    COLOR_GROUND_GRASS_DARK,
    COLOR_PIPE_BODY,
    COLOR_PIPE_DARK,
    COLOR_PIPE_LIGHT,
    COLOR_PIPE_BORDER,
    COLOR_BIRD_BODY,
    COLOR_BIRD_BELLY,
    COLOR_BIRD_WING,
    COLOR_BIRD_BEAK,
    COLOR_BIRD_EYE,
    COLOR_BIRD_PUPIL,
    COLOR_WHITE,
    COLOR_BLACK,
    COLOR_SHADOW,
)
from src.game.bird import Bird
from src.game.pipe import PipePair
from src.game.ground import Ground


class FlappyRenderer:
    """Renders game visuals to a Pygame display or off-screen surface."""

    def __init__(self, surface: pygame.Surface | None = None):
        if surface is not None and not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.surface = surface if surface is not None else pygame.Surface((self.width, self.height))

        # Fonts
        self.font_score = pygame.font.SysFont("Arial", 40, bold=True)
        self.font_title = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_msg = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_debug = pygame.font.SysFont("Courier", 14)

    def draw_background(self) -> None:
        """Draw sky and decorative backdrop (clouds, skyline)."""
        self.surface.fill(COLOR_SKY)

        # Draw decorative background clouds
        for cx, cy, rad in [(60, 100, 24), (85, 95, 30), (115, 100, 24),
                            (260, 160, 22), (285, 152, 28), (315, 160, 22)]:
            pygame.draw.circle(self.surface, (230, 245, 250), (cx, cy), rad)
            pygame.draw.circle(self.surface, (255, 255, 255), (cx, cy - 2), rad - 2)

        # Distant hills silhouette
        hills = [
            (0, GROUND_Y),
            (50, GROUND_Y - 45),
            (140, GROUND_Y - 15),
            (230, GROUND_Y - 55),
            (330, GROUND_Y - 20),
            (400, GROUND_Y - 40),
            (400, GROUND_Y),
        ]
        pygame.draw.polygon(self.surface, (140, 215, 175), hills)

    def draw_pipe(self, pipe: PipePair) -> None:
        """Draw a single upper and lower pipe with collars and borders."""
        collar_h = 24
        collar_pad = 4  # Overhang on left and right

        # 1. Top pipe
        top_rect = pipe.get_top_rect()
        # Body
        body_h = max(0, int(top_rect.height - collar_h))
        if body_h > 0:
            body_rect = pygame.Rect(top_rect.x, 0, top_rect.width, body_h)
            pygame.draw.rect(self.surface, COLOR_PIPE_BODY, body_rect)
            # Highlights & Shadows
            pygame.draw.rect(self.surface, COLOR_PIPE_LIGHT, (body_rect.x + 4, 0, 8, body_h))
            pygame.draw.rect(self.surface, COLOR_PIPE_DARK, (body_rect.right - 8, 0, 8, body_h))
            pygame.draw.rect(self.surface, COLOR_PIPE_BORDER, body_rect, 3)

        # Collar (rim at the bottom of the top pipe)
        collar_rect = pygame.Rect(
            top_rect.x - collar_pad,
            max(0, top_rect.height - collar_h),
            top_rect.width + 2 * collar_pad,
            collar_h,
        )
        pygame.draw.rect(self.surface, COLOR_PIPE_BODY, collar_rect)
        pygame.draw.rect(self.surface, COLOR_PIPE_LIGHT, (collar_rect.x + 4, collar_rect.y, 8, collar_h))
        pygame.draw.rect(self.surface, COLOR_PIPE_DARK, (collar_rect.right - 8, collar_rect.y, 8, collar_h))
        pygame.draw.rect(self.surface, COLOR_PIPE_BORDER, collar_rect, 3)

        # 2. Bottom pipe
        bot_rect = pipe.get_bottom_rect()
        # Collar (rim at the top of bottom pipe)
        bot_collar = pygame.Rect(
            bot_rect.x - collar_pad,
            bot_rect.y,
            bot_rect.width + 2 * collar_pad,
            collar_h,
        )
        pygame.draw.rect(self.surface, COLOR_PIPE_BODY, bot_collar)
        pygame.draw.rect(self.surface, COLOR_PIPE_LIGHT, (bot_collar.x + 4, bot_collar.y, 8, collar_h))
        pygame.draw.rect(self.surface, COLOR_PIPE_DARK, (bot_collar.right - 8, bot_collar.y, 8, collar_h))
        pygame.draw.rect(self.surface, COLOR_PIPE_BORDER, bot_collar, 3)

        # Body
        bot_body_y = bot_rect.y + collar_h
        bot_body_h = max(0, bot_rect.height - collar_h)
        if bot_body_h > 0:
            bot_body_rect = pygame.Rect(bot_rect.x, bot_body_y, bot_rect.width, bot_body_h)
            pygame.draw.rect(self.surface, COLOR_PIPE_BODY, bot_body_rect)
            pygame.draw.rect(self.surface, COLOR_PIPE_LIGHT, (bot_body_rect.x + 4, bot_body_y, 8, bot_body_h))
            pygame.draw.rect(self.surface, COLOR_PIPE_DARK, (bot_body_rect.right - 8, bot_body_y, 8, bot_body_h))
            pygame.draw.rect(self.surface, COLOR_PIPE_BORDER, bot_body_rect, 3)

    def draw_ground(self, ground: Ground) -> None:
        """Draw scrolling ground with grass border and diagonal soil stripes."""
        # Base ground soil
        ground_rect = pygame.Rect(0, ground.y, self.width, ground.height)
        pygame.draw.rect(self.surface, COLOR_GROUND_BASE, ground_rect)

        # Diagonal stripes
        stripe_w = ground.stripe_width
        scroll = ground.scroll_x
        num_stripes = int(self.width / stripe_w) + 3
        for i in range(-1, num_stripes):
            x = int(i * stripe_w + scroll)
            points = [
                (x, ground.y + 12),
                (x + stripe_w // 2, ground.y + 12),
                (x - 6 + stripe_w // 2, self.height),
                (x - 6, self.height),
            ]
            pygame.draw.polygon(self.surface, COLOR_GROUND_STRIPE, points)

        # Grass strip on top of ground
        pygame.draw.rect(self.surface, COLOR_GROUND_GRASS, (0, ground.y, self.width, 12))
        pygame.draw.rect(self.surface, COLOR_GROUND_GRASS_DARK, (0, ground.y + 9, self.width, 3))
        pygame.draw.line(self.surface, COLOR_BLACK, (0, ground.y), (self.width, ground.y), 2)

    def draw_bird(self, bird: Bird) -> None:
        """Draw an animated, tilted Flappy Bird."""
        # Create a dedicated transparent surface for bird rotation
        bird_surf = pygame.Surface((bird.width + 12, bird.height + 12), pygame.SRCALPHA)
        center_x = (bird.width + 12) // 2
        center_y = (bird.height + 12) // 2

        # Bird body (oval)
        body_rect = pygame.Rect(center_x - 14, center_y - 10, 28, 20)
        pygame.draw.ellipse(bird_surf, COLOR_BIRD_BODY, body_rect)
        pygame.draw.ellipse(bird_surf, COLOR_BLACK, body_rect, 2)

        # Belly highlight
        belly_rect = pygame.Rect(center_x - 12, center_y - 4, 16, 12)
        pygame.draw.ellipse(bird_surf, COLOR_BIRD_BELLY, belly_rect)

        # Eye & Pupil
        eye_center = (center_x + 6, center_y - 4)
        pygame.draw.circle(bird_surf, COLOR_BIRD_EYE, eye_center, 5)
        pygame.draw.circle(bird_surf, COLOR_BLACK, eye_center, 5, 1)
        pygame.draw.circle(bird_surf, COLOR_BIRD_PUPIL, (eye_center[0] + 1, eye_center[1] - 1), 2)

        # Beak
        beak_pts = [
            (center_x + 9, center_y - 2),
            (center_x + 18, center_y + 1),
            (center_x + 9, center_y + 6),
        ]
        pygame.draw.polygon(bird_surf, COLOR_BIRD_BEAK, beak_pts)
        pygame.draw.polygon(bird_surf, COLOR_BLACK, beak_pts, 1)

        # Wing with flap animation offsets
        wing_y_offset = {0: 0, 1: -4, 2: 3}.get(bird.wing_state, 0)
        wing_rect = pygame.Rect(center_x - 12, center_y - 3 + wing_y_offset, 14, 9)
        pygame.draw.ellipse(bird_surf, COLOR_BIRD_WING, wing_rect)
        pygame.draw.ellipse(bird_surf, COLOR_BLACK, wing_rect, 1)

        # Rotate bird surface based on velocity
        rotated_surf = pygame.transform.rotate(bird_surf, bird.rotation)
        rot_rect = rotated_surf.get_rect(center=(bird.x + bird.width / 2, bird.y + bird.height / 2))
        self.surface.blit(rotated_surf, rot_rect)

    def draw_score(self, score: int) -> None:
        """Render retro drop-shadow score in the upper center."""
        text = str(score)
        # Shadow
        shadow_surf = self.font_score.render(text, True, COLOR_SHADOW)
        rect = shadow_surf.get_rect(center=(self.width // 2 + 2, 52))
        self.surface.blit(shadow_surf, rect)

        # Main text
        score_surf = self.font_score.render(text, True, COLOR_WHITE)
        rect = score_surf.get_rect(center=(self.width // 2, 50))
        self.surface.blit(score_surf, rect)

    def draw_start_screen(self) -> None:
        """Overlay instructions before the first flap."""
        title = self.font_title.render("FLAPPY BIRD", True, COLOR_WHITE)
        shadow = self.font_title.render("FLAPPY BIRD", True, COLOR_SHADOW)
        r_shadow = shadow.get_rect(center=(self.width // 2 + 2, 162))
        r_title = title.get_rect(center=(self.width // 2, 160))
        self.surface.blit(shadow, r_shadow)
        self.surface.blit(title, r_title)

        msg1 = self.font_msg.render("Press SPACE to Flap & Start", True, COLOR_WHITE)
        r_msg1 = msg1.get_rect(center=(self.width // 2, 230))
        msg2 = self.font_msg.render("Press ESC to Quit", True, (240, 240, 240))
        r_msg2 = msg2.get_rect(center=(self.width // 2, 260))
        self.surface.blit(msg1, r_msg1)
        self.surface.blit(msg2, r_msg2)

    def draw_game_over(self, score: int, high_score: int) -> None:
        """Overlay game-over banner with current and high score."""
        box_w, box_h = 280, 160
        box_x = (self.width - box_w) // 2
        box_y = (self.height - box_h) // 2 - 20

        # Semi-transparent overlay box
        overlay = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (0, 0, 0, 160), (0, 0, box_w, box_h), border_radius=12)
        pygame.draw.rect(overlay, COLOR_WHITE, (0, 0, box_w, box_h), width=2, border_radius=12)
        self.surface.blit(overlay, (box_x, box_y))

        # Title
        title = self.font_title.render("GAME OVER", True, (255, 80, 80))
        self.surface.blit(title, title.get_rect(center=(self.width // 2, box_y + 35)))

        # Scores
        info1 = self.font_msg.render(f"Score: {score}", True, COLOR_WHITE)
        self.surface.blit(info1, info1.get_rect(center=(self.width // 2, box_y + 75)))

        info2 = self.font_msg.render(f"Best: {high_score}", True, (255, 215, 0))
        self.surface.blit(info2, info2.get_rect(center=(self.width // 2, box_y + 105)))

        hint = self.font_debug.render("Press SPACE or R to Restart", True, (200, 200, 200))
        self.surface.blit(hint, hint.get_rect(center=(self.width // 2, box_y + 135)))

    def draw_debug(self, bird: Bird, pipes: list[PipePair], next_pipe: PipePair | None) -> None:
        """Draw collision boxes and feature guidance lines for debugging."""
        # Bird hitbox
        pygame.draw.rect(self.surface, (255, 0, 0), bird.get_hitbox(), 2)

        # Pipe hitboxes
        for p in pipes:
            pygame.draw.rect(self.surface, (255, 0, 0), p.get_top_rect(), 2)
            pygame.draw.rect(self.surface, (255, 0, 0), p.get_bottom_rect(), 2)

        # Highlight next pipe target
        if next_pipe is not None:
            # Draw line to gap center
            gap_center_y = int(next_pipe.gap_top + next_pipe.gap_size / 2)
            pygame.draw.line(
                self.surface,
                (255, 255, 0),
                (int(bird.x + bird.width), int(bird.y + bird.height / 2)),
                (int(next_pipe.x), gap_center_y),
                2,
            )

    def get_rgb_array(self) -> np.ndarray:
        """Returns 3D numpy array [height, width, 3] of the current frame in RGB format."""
        # pygame.surfarray returns array indexed [width, height, channels]
        array_3d = pygame.surfarray.array3d(self.surface)
        return np.transpose(array_3d, (1, 0, 2))

