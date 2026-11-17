"""Play Flappy Bird by hand (Checkpoint 1).

Controls:
  SPACE / UP / Left Mouse Click: Flap (or Start / Restart)
  R: Restart immediately
  D: Toggle debug hitboxes & target lines
  ESC: Exit game
"""

import argparse
import sys
from src.game.constants import PIPE_GAP_SIZE, PIPE_SPEED
from src.game.flappy_game import FlappyBirdGame


def main():
    parser = argparse.ArgumentParser(description="Playable Flappy Bird by Hand (Checkpoint 1)")
    parser.add_argument("--gap-size", type=int, default=PIPE_GAP_SIZE, help="Vertical gap between pipes (default: 140)")
    parser.add_argument("--speed", type=float, default=PIPE_SPEED, help="Horizontal scrolling speed of pipes (default: 3.5)")
    parser.add_argument("--debug", action="store_true", help="Start with debug hitboxes and guide lines enabled")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible pipe generation")

    args = parser.parse_args()

    game = FlappyBirdGame(
        pipe_gap_size=args.gap_size,
        pipe_speed=args.speed,
        seed=args.seed,
    )
    if args.debug:
        game.show_debug = True

    print("=" * 60)
    print(" FLAPPY BIRD - CHECKPOINT 1: GAME FOUNDATION")
    print("=" * 60)
    print(" Controls:")
    print("   [SPACE] / [UP] / [Left Click] : Flap / Start / Restart")
    print("   [R]                           : Quick restart")
    print("   [D]                           : Toggle debug hitboxes & targets")
    print("   [ESC]                         : Quit game")
    print("=" * 60)

    try:
        game.run_interactive()
    except KeyboardInterrupt:
        print("\nExiting Flappy Bird. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()

