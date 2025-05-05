import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A slalom of alternating, raised beams: tests lateral stepping, precision foot placement, and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    ##############################
    # Course Description
    # The course is a series of narrow (40-50cm) beams, each 1.5m long, spaced 1m apart.
    # Beams alternate left and right from the central line, requiring the robot to "slalom" 
    # and turn at each goal, walking across each beam in sequence. 
    # The beams are raised above a shallow "moat" (depressed floor).
    # This tests turning, careful foot placement, and precision lateral control.
    ##############################

    num_beams = 6                # Beams + final goal = 7
    beam_width = 0.45 + 0.05 * (1 - difficulty)   # 0.5m (easy) ... 0.45