import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Repeating zig-zag balance beams: robot must cross a series of narrow, angled beams above pits to test balance and precise turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))
    course_len_idx = m_to_idx(length)
    course_wid_idx = m_to_idx(width)

    # Parameters for beams and pits
    beam_width_m = 0.45 + 0.1 * (1 - difficulty)      # 0.45m at hard, up to 0.55m at easy
    beam_length_m = 2.1 - 0.5 * difficulty            # Shorter at higher diff (2.1m->1.6m)
    pit_length_m = 0.8 + 0.6 * difficulty             # Longer pits at harder levels 
    pit_depth = 0.0 - (0.16 + 0.28 * difficulty)      # Pits get deeper (down to -0.