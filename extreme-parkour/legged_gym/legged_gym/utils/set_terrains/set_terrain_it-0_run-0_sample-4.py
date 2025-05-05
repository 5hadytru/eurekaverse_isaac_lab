import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone course: Series of narrow and wide elevated 'stones' and gaps forcing precise foot placement and short jumps."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Center line along the course, where stones will appear
    center_y = m_to_idx(width) // 2

    # Stepping stone parameters scale with difficulty
    min_stone_length = 0.5     # min stone "length" (along x), meters
    max_stone_length = 1.2     # max stone "length", meters
    min_stone_width = 0.45 if difficulty < 0.5 else 0.55 - 0.15 * difficulty # meters (must always be >= 0.4)
    max_stone_width = 1.4      # meters
    
    min_gap = 0.15 + 0.3 * difficulty     # Minimum gap between stones, meters
    max_gap = 0