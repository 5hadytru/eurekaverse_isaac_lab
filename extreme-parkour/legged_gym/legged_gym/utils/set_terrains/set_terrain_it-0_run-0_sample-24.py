import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone bridge and balance beam: Robot must cross a zig-zag series of narrow stepping stones followed by a long, narrow balance beam spanning a pit."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # ======== Course Parameters ========
    # Stepping stones parameters
    num_stepping_stones = 4
    stone_size_min = 0.45   # for hardest case, barely wider than quadruped stance
    stone_size_max = 0.9    # at easy, roomy stones
    stone_size = stone_size_max - (stone_size_max-stone_size_min)*difficulty
    # Stones must have at least minimal safe distance in-between
    stone_gap_min = 0.18
    stone_gap_max = 0.55
    stone_gap = stone_gap_min + (stone_gap_max-stone_gap_min)*difficulty

    # All stones at same height
    stone_height_min = 0.02