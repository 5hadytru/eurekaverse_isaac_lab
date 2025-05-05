import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Alternating side ramps ('A-frames') for testing high step and sloped walking skills."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Robot spawn coordinates
    spawn_x = m_to_idx(1.0)
    spawn_y = m_to_idx(width / 2)

    # Define ramp parameters (ramps alternate left-right along the corridor)
    ramp_base_length = 1.2 + 0.8 * difficulty     # Base length increases slightly with difficulty
    ramp_top_width = 0.6 - 0.25 * difficulty      # The flat top gets narrower at higher difficulty, min 0.35 m
    ramp_height = 0.18 + 0.15 * difficulty        # Higher ramps are harder
    gap_between_ramps = 0.6 - 0.25 * difficulty   # Short gaps for stepping, tighter spacing as difficulty rises

    ramp_base_length = m_to_idx(ramp_base_length