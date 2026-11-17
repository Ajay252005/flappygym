"""Gymnasium-compliant environment wrapper for the Flappy Bird game."""

from typing import Any
import gymnasium as gym
from gymnasium import spaces
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
    MAX_DROP_SPEED,
    FPS,
)
from src.game.flappy_game import FlappyBirdGame


class FlappyBirdEnv(gym.Env):
    """Custom Gymnasium environment for Flappy Bird.

    Action Space:
        Discrete(2):
            0: Do nothing (let gravity pull the bird downward)
            1: Flap (apply instant upward vertical impulse)

    Observation Space:
        Box:
        If reduced_observation is False (5 features):
            0: Bird y position [px]
            1: Bird vertical velocity [px/frame]
            2: Horizontal distance to next pipe [px]
            3: Next pipe gap top position [px]
            4: Next pipe gap bottom position [px]
        If reduced_observation is True (4 features):
            0: Bird y position [px]
            1: Horizontal distance to next pipe [px]
            2: Next pipe gap top position [px]
            3: Next pipe gap bottom position [px]

        If normalize_obs is True:
            Coordinates and velocities are scaled to roughly [-1, 1] or [0, 1].

    Rewards:
        Dense mode (default):
            +0.1 per frame survived
            +5.0 per pipe cleared
            -5.0 on collision / termination
        Sparse mode:
            +1.0 per pipe cleared
            0.0 otherwise
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        reward_mode: str = "dense",
        reduced_observation: bool = False,
        normalize_obs: bool = False,
        survival_reward: float = 0.1,
        pipe_reward: float = 5.0,
        death_penalty: float = -5.0,
        pipe_gap_size: int = PIPE_GAP_SIZE,
        pipe_speed: float = PIPE_SPEED,
        pipe_spawn_interval: int = PIPE_SPAWN_INTERVAL,
        gravity: float = GRAVITY,
        flap_strength: float = FLAP_STRENGTH,
        max_episode_steps: int | None = None,
    ):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"Invalid render_mode: {render_mode}. Valid modes: {self.metadata['render_modes']}"
        )
        assert reward_mode in ("dense", "sparse"), (
            f"Invalid reward_mode: {reward_mode}. Must be 'dense' or 'sparse'."
        )

        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.reduced_observation = reduced_observation
        self.normalize_obs = normalize_obs
        self.max_episode_steps = max_episode_steps

        # Reward shaping parameters
        self.survival_reward = survival_reward if reward_mode == "dense" else 0.0
        self.pipe_reward = pipe_reward if reward_mode == "dense" else 1.0
        self.death_penalty = death_penalty if reward_mode == "dense" else 0.0

        # Game configuration
        self.pipe_gap_size = pipe_gap_size
        self.pipe_speed = pipe_speed
        self.pipe_spawn_interval = pipe_spawn_interval
        self.gravity = gravity
        self.flap_strength = flap_strength

        # Action space: 0 = idle, 1 = flap
        self.action_space = spaces.Discrete(2)

        # Observation space bounds
        if not self.reduced_observation:
            # 5 features
            if self.normalize_obs:
                low = np.array([-0.2, -2.5, -0.5, 0.0, 0.0], dtype=np.float32)
                high = np.array([1.2, 2.5, 2.0, 1.0, 1.0], dtype=np.float32)
            else:
                low = np.array([-50.0, -25.0, -100.0, 0.0, 0.0], dtype=np.float32)
                high = np.array(
                    [
                        float(SCREEN_HEIGHT + 50.0),
                        25.0,
                        float(SCREEN_WIDTH + 300.0),
                        float(SCREEN_HEIGHT),
                        float(SCREEN_HEIGHT),
                    ],
                    dtype=np.float32,
                )
        else:
            # 4 features (ablation: omit velocity)
            if self.normalize_obs:
                low = np.array([-0.2, -0.5, 0.0, 0.0], dtype=np.float32)
                high = np.array([1.2, 2.0, 1.0, 1.0], dtype=np.float32)
            else:
                low = np.array([-50.0, -100.0, 0.0, 0.0], dtype=np.float32)
                high = np.array(
                    [
                        float(SCREEN_HEIGHT + 50.0),
                        float(SCREEN_WIDTH + 300.0),
                        float(SCREEN_HEIGHT),
                        float(SCREEN_HEIGHT),
                    ],
                    dtype=np.float32,
                )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Initialize internal game engine
        self.game = FlappyBirdGame(
            pipe_gap_size=self.pipe_gap_size,
            pipe_speed=self.pipe_speed,
            pipe_spawn_interval=self.pipe_spawn_interval,
            gravity=self.gravity,
            flap_strength=self.flap_strength,
            headless=(render_mode != "human"),
        )

        # Pygame display for human mode
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None

    def _get_obs(self) -> np.ndarray:
        """Extract and format the observation vector."""
        raw_obs = self.game.get_observation(reduced=self.reduced_observation)

        if self.normalize_obs:
            if not self.reduced_observation:
                norm_obs = np.array(
                    [
                        raw_obs[0] / SCREEN_HEIGHT,
                        raw_obs[1] / MAX_DROP_SPEED,
                        raw_obs[2] / SCREEN_WIDTH,
                        raw_obs[3] / SCREEN_HEIGHT,
                        raw_obs[4] / SCREEN_HEIGHT,
                    ],
                    dtype=np.float32,
                )
            else:
                norm_obs = np.array(
                    [
                        raw_obs[0] / SCREEN_HEIGHT,
                        raw_obs[1] / SCREEN_WIDTH,
                        raw_obs[2] / SCREEN_HEIGHT,
                        raw_obs[3] / SCREEN_HEIGHT,
                    ],
                    dtype=np.float32,
                )
            return np.clip(norm_obs, self.observation_space.low, self.observation_space.high)

        return np.clip(raw_obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        """Return diagnostic info dictionary."""
        return {
            "score": int(self.game.score),
            "pipes_passed": int(self.game.score),
            "frames_survived": int(self.game.frames_survived),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)

        self.game.reset(seed=seed)

        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird RL Environment")
            self.clock = pygame.time.Clock()

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one action step in the environment.

        Args:
            action: 0 (idle) or 1 (flap)

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        assert self.action_space.contains(action), f"Invalid action: {action}"

        is_game_over, pipes_passed = self.game.step(action)

        # Reward computation
        reward = float(self.survival_reward)
        if self.reward_mode == "dense":
            # Reward shaping: encourage maintaining height near upcoming gap center
            next_pipe = self.game.get_next_pipe()
            if next_pipe is not None:
                gap_center = next_pipe.gap_top + next_pipe.gap_size / 2.0
                dist_norm = abs(self.game.bird.y - gap_center) / (SCREEN_HEIGHT / 2.0)
                reward += float(max(0.0, 0.05 * (1.0 - dist_norm)))

        if pipes_passed > 0:
            reward += float(pipes_passed * self.pipe_reward)
        if is_game_over:
            reward += float(self.death_penalty)

        terminated = bool(is_game_over)
        truncated = False
        if self.max_episode_steps is not None and self.game.frames_survived >= self.max_episode_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the environment based on render_mode."""
        if self.render_mode == "rgb_array":
            return self.game.render(mode="rgb_array")
        elif self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Flappy Bird RL Environment")
                self.clock = pygame.time.Clock()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise KeyboardInterrupt("Pygame window closed by user.")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.close()
                        raise KeyboardInterrupt("Evaluation stopped by user (ESC).")
                    elif event.key == pygame.K_d:
                        self.game.show_debug = not self.game.show_debug

            self.game.render(mode="human", screen=self.screen)
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
            return None
        return None

    def close(self) -> None:
        """Clean up Pygame resources."""
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None
            self.clock = None

