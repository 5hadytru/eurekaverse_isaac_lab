import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone course: quadruped must traverse a series of small, widely spaced platforms (stepping stones) across a wide pit to test precision stepping and short, accurate jumps."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Set spawn area to flat ground
    spawn_length = m_to_idx(2)
    height_field[:spawn_length, :] = 0

    # Stepping stone parameters
    # Each "stone" is a small square platform, placed with gaps in between (all surrounded by a wide pit)
    # At low difficulty: stones are bigger, closer; at high difficulty: smaller, farther apart
    stone_side_min = 0.45 - 0.18 * difficulty  # meters (minimum allowed: 0.27 at difficulty=1, but clamp to 0.4)
    stone_side = max(0.4, stone_side_min)      # for safety and clarity, minimum 0.4