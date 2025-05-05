import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of low 'brick wall' step-overs for the quadruped to step or jump over, as seen in dog or military obstacle courses."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- Parameters for wall steps ---
    # Wall width: always at least 1m for stable crossing
    wall_width = 1.2
    wall_width_idx = m_to_idx(wall_width)
    
    # Wall thickness: make them "slabs" the quadruped can step over
    wall_thickness = 0.22  # well less than quadruped length
    wall_thickness_idx = m_to_idx(wall_thickness)
    
    # Wall height varies with difficulty: 0.12 to 0.34 meters (from low step to high, but below maximum lift)
    wall_h_min = 0.12 + 0.10 * difficulty
    wall_h_max = 0.22 +