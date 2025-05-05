import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Slalom-style course with 7 low, wide 'rails' that form an alternating zig-zag, requiring the quadruped to turn tightly and balance as it walks atop each rail."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Set up the rail ("beam") dimensions
    # Rail width varies slightly with difficulty (narrower/harder)
    rail_width_m = 1.3 - 0.7 * difficulty  # 1.3m (easy), 0.6m (hard)
    rail_length_m = 1.3 + 0.5 * difficulty  # 1.3m-1.8m
    rail_height = 0.09 + 0.07 * difficulty  # just off the ground, up to 0.16m

    gap_min = 0.15 + 0.15 * difficulty  # gap between rails, meters
    gap_max = 0.