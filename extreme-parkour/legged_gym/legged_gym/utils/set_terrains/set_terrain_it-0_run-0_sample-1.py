import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of low, wide see-saw tilting beams requiring careful balancing and foot placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return (np.round(m / field_resolution).astype(np.int16)
                if not (isinstance(m, list) or isinstance(m, tuple))
                else [round(i / field_resolution) for i in m])

    # Initialize flat ground
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course parameters (quantized)
    course_x = m_to_idx(length)
    course_y = m_to_idx(width)
    spawn_length = m_to_idx(2)
    min_beam_width = 1.0
    beam_width = min_beam_width + 0.3 * (1 - difficulty)
    beam_width_idx = m_to_idx(beam_width)
    beam_length = 1.2 + 0.6 * difficulty
    beam_length_idx = m_to_idx(beam_length)
    gap_length = 0.5 + 0.5 * difficulty
    gap_length_idx = m_to_idx(gap_length)
    beam_height_min = 0.12 + 0.10 * difficulty
    beam_height_max = 0.18