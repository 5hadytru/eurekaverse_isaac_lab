import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of narrow, staggered balance beams and zig-​zag path to test precise foot placement and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2)) # Each as (x, y) in idx units

    # Parameters
    mid_y = m_to_idx(width / 2)
    course_x = m_to_idx(length)
    course_y = m_to_idx(width)

    spawn_x = m_to_idx(2)
    beam_length = 1.8 - 0.3 * difficulty           # Beams get slightly shorter with difficulty
    beam_width = 0.47 - 0.25 * difficulty          # Beams get narrower with difficulty; never < 0.22m
    beam_width = max(beam_width, 0.22)
    beam_height = 0.08 + 0.20 * difficulty         # Higher with difficulty

    gap = 0.18 + 0.32 * difficulty                 # G