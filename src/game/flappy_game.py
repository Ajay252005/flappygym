"""Core Flappy Bird engine supporting both interactive human play and headless RL simulation."""

from enum import Enum
import random
import numpy as np
import pygame
from src.game.constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    PIPE_WIDTH,
    PIPE_GAP_SIZE,
    PIPE_SPEED,
    PIPE_SPAWN_INTERVAL,
    GRAVITY,
    FLAP_STRENGTH,
    FPS,
)
from src.game.bird import Bird
from src.game.pipe import PipePair
from src.game.ground import Ground
from src.game.renderer import FlappyRenderer


class GameState(Enum):
    START = "START"
    PLAYING = "PLAYING"
    GAME_OVER = "GAME_OVER"


class FlappyBirdGame:
    """Core game engine managing physics, obstacles, collisions, and state."""

    def __init__(
        self,
        pipe_gap_size: int = PIPE_GAP_SIZE,
        pipe_speed: float = PIPE_SPEED,
        pipe_spawn_interval: int = PIPE_SPAWN_INTERVAL,
        gravity: float = GRAVITY,
        flap_strength: float = FLAP_STRENGTH,
        seed: int | None = None,
        headless: bool = False,
    ):
        self.pipe_gap_size = pipe_gap_size
        self.pipe_speed = pipe_speed
        self.pipe_spawn_interval = pipe_spawn_interval
        self.gravity = gravity
        self.flap_strength = flap_strength
        self.headless = headless

        self.rng = random.Random(seed) if seed is not None else random.Random()

        self.bird = Bird(gravity=self.gravity, flap_strength=self.flap_strength)
        self.ground = Ground(speed=self.pipe_speed)
        self.pipes: list[PipePair] = []

        self.score = 0
        self.high_score = 0
        self.frames_survived = 0
        self.game_state = GameState.START
        self.collision_detected = False

        self.renderer: FlappyRenderer | None = None
        self.show_debug = False

    def reset(self, seed: int | None = None) -> None:
        """Reset game state to start a new round."""
        if seed is not None:
            self.rng = random.Random(seed)

        self.bird.reset()
        self.ground.reset()
        self.pipes.clear()

        # Spawn initial pipes ahead of the bird
        first_pipe_x = SCREEN_WIDTH + 80
        self.pipes.append(
            PipePair(
                x=first_pipe_x,
                gap_size=self.pipe_gap_size,
                speed=self.pipe_speed,
                rng=self.rng,
            )
        )
        self.pipes.append(
            PipePair(
                x=first_pipe_x + self.pipe_spawn_interval,
                gap_size=self.pipe_gap_size,
                speed=self.pipe_speed,
                rng=self.rng,
            )
        )

        self.score = 0
        self.frames_survived = 0
        self.game_state = GameState.PLAYING
        self.collision_detected = False

    def get_next_pipe(self) -> PipePair | None:
        """Return the nearest upcoming pipe pair ahead of or currently around the bird."""
        for pipe in self.pipes:
            if pipe.x + pipe.width >= self.bird.x:
                return pipe
        return None

    def get_observation(self, reduced: bool = False) -> np.ndarray:
        """Get low-dimensional state representation.

        Features:
          0: Bird height (y)
          1: Bird velocity (vy) [omitted if reduced=True]
          2: Horizontal distance to next pipe
          3: Next pipe gap top position
          4: Next pipe gap bottom position
        """
        next_pipe = self.get_next_pipe()
        if next_pipe is not None:
            dist_to_pipe = next_pipe.x - self.bird.x
            gap_top = next_pipe.gap_top
            gap_bottom = next_pipe.gap_bottom
        else:
            dist_to_pipe = float(SCREEN_WIDTH)
            gap_top = float(SCREEN_HEIGHT / 2 - self.pipe_gap_size / 2)
            gap_bottom = float(SCREEN_HEIGHT / 2 + self.pipe_gap_size / 2)

        if reduced:
            # Observation ablation: omit velocity
            return np.array(
                [self.bird.y, dist_to_pipe, gap_top, gap_bottom],
                dtype=np.float32,
            )
        return np.array(
            [self.bird.y, self.bird.velocity, dist_to_pipe, gap_top, gap_bottom],
            dtype=np.float32,
        )

    def step(self, action: int) -> tuple[bool, int]:
        """Advance game physics by one frame.

        action: 1 = flap, 0 = idle
        Returns: (is_game_over, pipes_passed_this_step)
        """
        if self.game_state == GameState.GAME_OVER:
            return True, 0

        # Action handling
        if action == 1:
            self.bird.flap()

        # Update entities
        self.bird.update()
        self.ground.update()

        for pipe in self.pipes:
            pipe.update()

        # Check pipe spawning: keep enough pipes ahead
        if self.pipes:
            rightmost_x = max(p.x for p in self.pipes)
            if rightmost_x < SCREEN_WIDTH + self.pipe_spawn_interval:
                self.pipes.append(
                    PipePair(
                        x=rightmost_x + self.pipe_spawn_interval,
                        gap_size=self.pipe_gap_size,
                        speed=self.pipe_speed,
                        rng=self.rng,
                    )
                )

        # Remove offscreen pipes
        self.pipes = [p for p in self.pipes if not p.is_offscreen()]

        # Score checking
        pipes_passed = 0
        for pipe in self.pipes:
            if not pipe.passed and (pipe.x + pipe.width) < self.bird.x:
                pipe.passed = True
                self.score += 1
                pipes_passed += 1
                if self.score > self.high_score:
                    self.high_score = self.score

        # Collision detection
        bird_rect = self.bird.get_hitbox()

        # Ground collision
        if self.ground.collides_with(bird_rect):
            self.collision_detected = True

        # Ceiling collision (prevent flying over pipes)
        if bird_rect.top <= 0:
            self.collision_detected = True

        # Pipe collision
        for pipe in self.pipes:
            if pipe.collides_with(bird_rect):
                self.collision_detected = True
                break

        self.frames_survived += 1

        if self.collision_detected:
            self.game_state = GameState.GAME_OVER
            return True, pipes_passed

        return False, pipes_passed

    def render(self, mode: str = "human", screen: pygame.Surface | None = None) -> np.ndarray | None:
        """Render the current frame.

        mode: 'human' displays to screen, 'rgb_array' returns np.ndarray
        """
        if self.renderer is None or (screen is not None and self.renderer.surface != screen):
            self.renderer = FlappyRenderer(surface=screen)

        self.renderer.draw_background()

        for pipe in self.pipes:
            self.renderer.draw_pipe(pipe)

        self.renderer.draw_ground(self.ground)
        self.renderer.draw_bird(self.bird)
        self.renderer.draw_score(self.score)

        if self.show_debug:
            self.renderer.draw_debug(self.bird, self.pipes, self.get_next_pipe())

        if self.game_state == GameState.START:
            self.renderer.draw_start_screen()
        elif self.game_state == GameState.GAME_OVER:
            self.renderer.draw_game_over(self.score, self.high_score)

        if mode == "rgb_array":
            return self.renderer.get_rgb_array()
        return None

    def run_interactive(self) -> None:
        """Launch interactive Pygame loop for human keyboard/mouse play."""
        pygame.init()
        pygame.display.set_caption("Flappy Bird - Checkpoint 1")
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        self.renderer = FlappyRenderer(surface=screen)

        self.game_state = GameState.START
        running = True

        while running:
            clock.tick(FPS)
            action = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_SPACE, pygame.K_UP):
                        if self.game_state == GameState.START:
                            self.reset()
                            action = 1
                        elif self.game_state == GameState.PLAYING:
                            action = 1
                        elif self.game_state == GameState.GAME_OVER:
                            self.reset()
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_d:
                        self.show_debug = not self.show_debug
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.game_state == GameState.START:
                        self.reset()
                        action = 1
                    elif self.game_state == GameState.PLAYING:
                        action = 1
                    elif self.game_state == GameState.GAME_OVER:
                        self.reset()

            if self.game_state == GameState.PLAYING:
                self.step(action)

            self.render(mode="human", screen=screen)
            pygame.display.flip()

        pygame.quit()

