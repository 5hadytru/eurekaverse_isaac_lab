import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A 'stepping stone' sequence of narrow beams crossing a water pit, testing balance and precise foot placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # COURSE DESIGN:
    # Most of course is a "pit" (negative heights) crossed by 6 closely-spaced narrow beams
    # Each beam is 1.2m long (along y), 0.22-0.4m wide (narrow), 0.2-0.4m high. 
    # For difficulty, decrease beam width and increase spacing.
    # The course tests the robot's ability to balance and precisely walk along/over beams.

    beam_length = 1.2                         # meters along y (width axis)
    beam_width = 0.4 - 0.18 * difficulty      # meters, minimum 0.22m at high difficulty
    beam_height = 0.20 + 0.20 * difficulty    #