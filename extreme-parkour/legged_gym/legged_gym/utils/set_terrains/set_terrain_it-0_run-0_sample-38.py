import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of balance beams that force the robot to walk steady and precise over narrow elevated paths."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Definitions based on difficulty for beams: get narrower/higher with more difficulty
    beam_width_min = 0.4                     # Absolute minimum width
    beam_width_max = 1.1                     # Maximum width
    beam_width = beam_width_max - (beam_width_max - beam_width_min) * difficulty

    beam_height_min = 0.07                   # Minimal rise above ground (meters)
    beam_height_max = 0.22                   # At max difficulty, beams are high
    beam_height = beam_height_min + (beam_height_max - beam_height_min) * difficulty

    pit_depth_min = -0.05                    # At easy, a pit that is barely below ground
    pit_depth_max = -0.35                    # At hard, a deep pit around beams
    pit_depth =