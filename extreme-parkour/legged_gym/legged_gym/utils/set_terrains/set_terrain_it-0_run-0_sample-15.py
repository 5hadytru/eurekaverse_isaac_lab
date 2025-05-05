import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of slatted balance beams with variable spacing and slight turns, testing agility and precise paw placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # -- COURSE PARAMETERS --
    # Balance beam dimensions
    beam_length = 1.8 + 0.7 * difficulty     # longer beams on harder difficulty
    beam_length = min(beam_length, length - 2)  # fit within bounds
    beam_width = 1.0 - 0.5 * difficulty
    beam_width = max(beam_width, 0.4)        # never narrower than 0.4m
    beam_height = 0.12 + 0.18 * difficulty   # beams higher above ground as difficulty increases
    gap = 0.18 + 0.55 * difficulty           # spacing between beams increases
    num_beams = 6

    # Pit settings
    pit_height = -0.9 - 0