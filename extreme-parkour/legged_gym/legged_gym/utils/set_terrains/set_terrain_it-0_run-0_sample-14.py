import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of alternating raised balance beams and low steps spanning the arena width to test narrow traversal and step negotiation."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return (np.round(m / field_resolution).astype(np.int16)
                if not (isinstance(m, list) or isinstance(m, tuple)) 
                else [round(i / field_resolution) for i in m])

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    course_len_idx = m_to_idx(length)
    course_wid_idx = m_to_idx(width)

    # Terrain parameters
    spawn_len = 2.0
    balance_beam_length = 1.8 - 0.5 * difficulty      # meter
    balance_beam_width = 0.4                         # narrow, but crossable by robot, meters
    balance_beam_height = 0.15 + 0.15*difficulty     # up to 30cm
    beam_gap = 0.6 + 0.3 * difficulty                # gap between obstacles
    step_width = 1.3 + 0.3 * (1-difficulty)          # steps wider at low difficulty
    step_height = 0.06 +