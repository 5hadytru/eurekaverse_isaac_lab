import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Slalom course of offset, wide cylindrical 'logs' to test stepping over obstacles and tight turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course Parameters
    log_radius = 0.18 + 0.18 * difficulty  # 0.18–0.36 m, always much wider than the robot's body
    log_height = 0.1 + 0.18 * difficulty   # 0.1–0.28 m
    log_length = 1.9 + 0.7 * difficulty    # 1.9–2.6 m long; always longer than robot and 1m requirement
    log_radius_idx = m_to_idx(log_radius)
    log_length_idx = m_to_idx(log_length)
    log_height_scalar = log_height
    
    # Offset parameters
    lateral_span = width - 1.2         # How far logs can be offset laterally (so logs never touch edge