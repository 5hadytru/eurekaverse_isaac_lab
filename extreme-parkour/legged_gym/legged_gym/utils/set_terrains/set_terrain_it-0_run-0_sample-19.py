import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of sloped ramps alternating left/right, testing traversing sloped terrain and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for ramp obstacles
    n_ramps = 6
    ramp_length_m = 1.1 - 0.2 * difficulty            # ramps are slightly shorter at high difficulty
    ramp_width_m = 1.25                               # wide enough for the robot to turn on
    flat_segment_m = 0.25                             # flat space between ramps
    ramp_height_m = 0.18 + 0.22 * difficulty          # ramp height increases with difficulty

    # Convert to indices
    ramp_length = m_to_idx(ramp_length_m)
    ramp_width = m_to_idx(ramp_width_m)
    flat_segment = m_to_idx(flat_segment_m)
    ramp_height = ramp_height_m

    mid_y = m_to_idx(width/2)

    # Alternating ramps: each goes