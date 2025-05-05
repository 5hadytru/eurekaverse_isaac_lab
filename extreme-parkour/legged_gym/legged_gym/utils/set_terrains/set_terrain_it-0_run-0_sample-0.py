import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A balance beam course with alternating beam directions to test lateral stepping and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course parameters
    spawn_length = m_to_idx(2.0)
    course_length_idx = m_to_idx(length)
    course_width_idx = m_to_idx(width)

    # Balance beam parameters
    # Narrower beam and bigger gaps at higher difficulty; beams always >= 0.4m wide/long
    beam_width = 0.3 + (0.2 * (1-difficulty))           # 0.5m wide at easiest, 0.3m at hardest
    beam_width_idx = m_to_idx(beam_width)
    beam_height = 0.05 + 0.18 * difficulty             # 0.05m height at easy, up to 0.23m at hard
    gap_length = 0.5 + 0.8*difficulty                  # 0