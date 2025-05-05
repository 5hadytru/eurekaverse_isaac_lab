import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone pathway: robot must precisely walk along raised blocks with variable spacing and direction."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Terrain parameters
    spawn_length = m_to_idx(2)
    field_length_idx = m_to_idx(length)
    field_width_idx = m_to_idx(width)
    min_block_size = 0.4            # Smallest allowed block side (must fit quadruped)
    max_block_size = 0.7 + difficulty * 0.3  # Max block size increases with difficulty for some freedom
    min_block_size_idx = m_to_idx(min_block_size)
    max_block_size_idx = m_to_idx(max_block_size)
    stepping_height = 0.09 + 0.23 * difficulty  # Block height
    gap_size = 0.3 + (0.45 * difficulty)        # Gap between stepping stones increases with difficulty
    gap_size_idx = m_to_idx(gap_size