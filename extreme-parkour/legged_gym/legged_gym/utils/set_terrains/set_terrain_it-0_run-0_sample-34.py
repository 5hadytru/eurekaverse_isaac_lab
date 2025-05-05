import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of sloped ramps and steps to test the quadruped's ability to ascend, descend, and traverse inclined planes."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Course dimensions in quantized indices
    h_len, h_wid = m_to_idx(length), m_to_idx(width)
    height_field = np.zeros((h_len, h_wid))
    goals = np.zeros((8, 2))

    mid_y = h_wid // 2

    # Parameters for ramps & steps based on difficulty
    # All obstacles start after the safe spawn area (to x >= 2.0m)
    spawn_len = m_to_idx(2.0)
    height_field[:spawn_len, :] = 0
    
    # First goal: at the end of spawn area
    goals[0] = [spawn_len - m_to_idx(0.5), mid_y]

    # Ramp parameters (increase slope and height with difficulty)
    ramp_length = 2.2 - 1.0 * difficulty        # [2.2, 1.2