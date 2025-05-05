import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A 'slalom' of alternating staggered beams requiring precision foot placement and lateral maneuvering."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        if isinstance(m, (list, tuple, np.ndarray)):
            return [round(i / field_resolution) for i in m]
        return np.round(m / field_resolution).astype(np.int16)

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Define beam obstacle dimensions
    # Each beam: 2.3-2.8m long (for difficulty), width 0.38-0.55m, height 0.12-0.33m depending on difficulty
    beam_base_length = 2.3 + 0.5 * difficulty    # in meters
    beam_width = 0.38 + 0.17 * difficulty        # in meters
    beam_height = 0.12 + 0.21 * difficulty       # in meters

    # Define spacing between beams and stagger offset
    # Each new beam is offset sideways (left, right alternately)
    gap_between_beams = 0.6 + 0.4 * difficulty   # meters between beam ends
    stagger_offset = 0.65