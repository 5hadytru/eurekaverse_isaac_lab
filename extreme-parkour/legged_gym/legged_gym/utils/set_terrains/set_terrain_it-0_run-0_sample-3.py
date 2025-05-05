import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of tall 'stepping stone' blocks for the quadruped to traverse, testing precise foot placement and hopping/jumping ability."""
    
    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course parameters
    np.random.seed(42)  # Deterministic for reproducibility, remove or change in deployment
    num_blocks = 7
    block_size_mean = 0.6 + 0.2 * (1 - difficulty)        # Block size (meters), shrinks a bit as difficulty increases
    block_size_var = 0.08 + 0.08 * difficulty             # More variation at higher difficulties
    gap_min = 0.35 + 0.25 * difficulty                    # Minimum gap between stones increases with difficulty
    gap_max = 0.55 + 0.5 * difficulty                     # Maximum gap between stones increases further
    block_height = 0.10 + 0.20 * difficulty