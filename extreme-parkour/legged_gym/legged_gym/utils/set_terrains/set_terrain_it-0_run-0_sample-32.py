import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of low rails (balance beams) with turns: Tests agile, precise balancing and steering."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for beam ("rail") obstacles
    rail_width = 0.22 + 0.18 * (1 - difficulty)     # Between 0.22m (hard) and 0.4m (easiest)
    rail_height = 0.06 + 0.14 * difficulty          # 0.06m (easy) to 0.2m (hard)
    rail_length = 2.3 + 1.0 * (1 - difficulty)      # 2.3m (hard) to 3.3m (easy) - fits up to 3 turns nicely
    buffer = 0.08                                   # buffer between rails and boundaries, meters

    rail_width_idx = m_to_idx(rail_width)
    rail_height = float(rail