import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stones: Repeated narrow, elevated stepping stones crossing a water pit, testing the quadruped's precise foot placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Basic params
    length_idx = m_to_idx(length)
    width_idx = m_to_idx(width)
    spawn_x = m_to_idx(1)
    mid_y  = width_idx // 2

    # Course design: Stepping stones above a "pit" (e.g. water or mud).
    # -- All terrain except stones is negative/low height ("pit").
    # -- Stones are narrow, elevated platforms, with the distance and height rising with difficulty.
    stone_length = 0.52 + 0.18 * (1 - difficulty)  # shrinking as difficulty increases (min: 0.52m, max: 0.7m)
    stone_width = 0.4 + 0.2 * (1 - difficulty