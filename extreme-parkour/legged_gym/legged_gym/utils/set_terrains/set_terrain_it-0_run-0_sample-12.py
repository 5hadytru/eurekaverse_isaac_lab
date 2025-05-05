import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of alternating low and high balance beams with forced turns (slalom), testing turning while balancing."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Setup: Balance beams with alternating offsets and heights
    # Beams: 0.5 to 0.7m wide, 1.3-1.7m long (varied slightly as function of difficulty, still always >= 1m wide)
    # Beams have negative space (pit) on either side, so robot must stay on beam, not fall off.
    # Each beam is offset left or right to force turning.
    # Use at least 1.2m space between beams for turning zone.
    # Beams start at x=2; spawn area at x<2 is left flat.

    beam_width = 0.5 + 0.2 * difficulty  # 0.5~0.7m wide
    beam_length = 1.