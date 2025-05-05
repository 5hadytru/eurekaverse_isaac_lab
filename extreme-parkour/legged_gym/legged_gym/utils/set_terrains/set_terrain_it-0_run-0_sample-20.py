import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of 'stepping beams': parallel beams/poles to test narrow-foot placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for beam obstacles
    n_beams = 5 + int(difficulty * 2)  # 5 to 7 beams as difficulty rises
    beam_length = 1.8 - 0.4 * difficulty                    # meters, beams slightly shorter on high difficulty
    beam_width = 0.22 + 0.18*(1-difficulty)                 # meters, gets narrower with difficulty (0.22 to 0.4)
    beam_height = 0.10 + 0.15*difficulty                    # meters, beams are taller with difficulty
    beam_gap = 0.5 + 0.4*difficulty                         # meters between beams
    beam_y_center = m_to_idx(width/2)
    spawn_clear_x = m_to_idx(2.0)