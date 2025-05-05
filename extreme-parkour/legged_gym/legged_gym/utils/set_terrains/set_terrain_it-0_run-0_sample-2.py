import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Slalom-style alternating low rails to test precise lateral stepping and underfoot clearance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        if isinstance(m, list) or isinstance(m, tuple):
            return [round(i / field_resolution) for i in m]
        return np.round(m / field_resolution).astype(np.int16)

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Settings
    rail_length = 0.8 + 0.6 * difficulty  # rails get longer/harder
    rail_width = 0.2 + 0.2 * (1-difficulty)  # a little easier at low difficulty
    rail_width = max(rail_width, 0.2)  # but never less than 0.2m
    rail_height = 0.06 + 0.14 * difficulty  # up to 20cm for max
    rail_gap = 1.0 - 0.3 * difficulty  # vertical spacing between rails
    num_rails = 6  # always 6 rails

    y_clear = 0.5  # always keep 0.5m buffer on sides
    min_pad_from_spawn = 2.