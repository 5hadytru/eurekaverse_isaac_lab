import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of staggered balance beams to test the robot's narrow-footing and steering skill."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters: balance beam length, width, height, spacing
    beam_length = 1.8 - 0.5 * difficulty             # Beams get shorter on higher difficulty
    beam_width = 0.45 - 0.10 * difficulty            # Beams get narrower on higher difficulty (min: 0.35m)
    beam_height = 0.15 + 0.15 * difficulty           # Beams get higher
    gap_length = 0.4 + 0.8 * difficulty              # Gaps between beams get wider

    offset_angle = np.radians(12 + 13 * difficulty)  # Max angle for staggering beams

    beam_length_idx = m_to_idx(beam_length)
    beam_width_idx = max(m_to_idx(beam_width), m_to_idx