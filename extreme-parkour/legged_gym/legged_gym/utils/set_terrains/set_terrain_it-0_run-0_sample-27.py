import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping Stones: Sequential, slightly wobbly stepping stone tiles over a pit for precise foot placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    ### Terrain design setup ###
    # Main challenge: Walk across a narrow, single-tile-wide row (precise placement/"stepping stones") over a pit.
    # Each "stone" = a safe area ~0.5-0.7 m in length and width, separated by a gap.
    # The "stones" get smaller and gaps get larger as difficulty increases.

    n_stones = 7  # Number of stepping stones (and total 8 waypoints)
    pit_depth = -1.0  # meters, enough to penalize falling
    min_stone_size = 0.45     # stone is never smaller than robot's width plus buffer
    max_stone_size = 1.0
    stone_size = max_stone_size - (max_stone_size