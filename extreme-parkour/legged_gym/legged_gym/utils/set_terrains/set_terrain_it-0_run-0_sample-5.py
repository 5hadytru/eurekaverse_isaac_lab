import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Alternating 'A-frame ramps' and 'stairs' with flat landing zones in a zig-zag, to test climbing and controlled descents under turns."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Terrain parameters
    course_x = m_to_idx(length)
    course_y = m_to_idx(width)

    spawn_x = m_to_idx(1)
    center_y = course_y // 2
    margin_y = m_to_idx(0.8)   # minimum margin from either side for obstacles

    # Obstacle sizing (relative to quadruped)
    ramp_long = m_to_idx(1.2 + 0.4 * difficulty)   # ramp length increases with difficulty
    ramp_high = 0.12 + 0.18 * difficulty           # ramp max height (meters)
    flat_zone_length = m_to_idx(1.3 - 0.4 * difficulty)
    stairs_steps = 3 + int(