

import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone parkour: a series of narrow, spaced pillars ("stepping stones") over 'water', testing lateral balance and precise walking."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        if isinstance(m, (list, tuple, np.ndarray)):
            return [int(round(i / field_resolution)) for i in m]
        return int(round(m / field_resolution))

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Quadruped: 0.645 x 0.28