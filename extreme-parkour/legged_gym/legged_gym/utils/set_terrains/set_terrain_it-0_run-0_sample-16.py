import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of tilted balance beams that snake through the environment to test balance, precise foot placement, and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))
    
    # Skill tested: balance and precise stepping with turning
    
    # Parameters
    N_BEAMS = 6  # Number of beams/segments (plus start and end pads)
    beam_length_mean = 1.6 - 0.4 * difficulty    # Each beam length
    beam_length_var = 0.2 + 0.2 * difficulty
    beam_width = 0.45 + 0.10 * (1 - difficulty)  # slightly wider at low difficulty
    pad_length = 0.5
    pad_width = 1.2
    beam_height_variation = 0.03 + 0.10 * difficulty  # Each beam is slightly tilted, more so with higher difficulty
    beam_height_base = 0.05 + 0