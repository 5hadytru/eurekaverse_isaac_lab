import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Slalom of alternating ramps and low hurdles for quadruped balance and agile turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for obstacles
    ramp_length = 1.5 - 0.5 * difficulty       # Ramps get shorter and steeper with difficulty
    ramp_height = 0.10 + 0.25 * difficulty     # Ramps get steeper with difficulty
    ramp_width = 1.2                           # Fixed ramp width

    hurdle_length = 0.25 + 0.20 * difficulty   # Hurdles get a bit thicker as they get higher/harder
    hurdle_width = 1.0 + 0.2 * (1-difficulty)
    hurdle_height = 0.08 + 0.14 * difficulty   # Hurdles higher with difficulty

    track_center = m_to_idx(width/2.0)
    track_halfwidth = m_to_idx(0.