"""Pytest configuration ensuring the project root is always on sys.path."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

