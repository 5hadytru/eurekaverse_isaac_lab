import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping-stone pillars: the quadruped must traverse a series of round raised stepping stones over a pit, testing agile foot placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course Center
    mid_y = m_to_idx(width // 2)

    # Set up spawn flat area
    spawn_length = m_to_idx(2)
    height_field[0:spawn_length, :] = 0
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    # The pit (everything after the spawn is -1.0m)
    height_field[spawn_length:, :] = -1.0

    # Stepping stone parameters (all stones are set on top of the pit)
    # Pillar (stone) diameter decreases and spacing increases with difficulty
    min_diam, max_diam = 0.55, 0.75
    stone_diam = max