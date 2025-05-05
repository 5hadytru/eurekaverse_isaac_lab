import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of ascending and descending ramps with narrow flat tops to test precise foot placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # === Set up terrain and goals ===
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course design parameters
    # We'll make a zig-zag with multiple "balance beams": narrow platforms, each reached by going up and down ramps.
    # Each segment will be: ramp up, balance beam, ramp down, turn (L or R), repeat
    # Difficulty increases ramp angle and narrows the beams.

    total_segments = 4
    segment_length = (length - 2.0) / total_segments                    # meters, leave 2m for spawn/finish area
    ramp_ratio = 0.28 + 0.32 * difficulty                               # fraction of segment taken by ramp
    beam_ratio = 0.45 - 0.25 * difficulty                               # fraction of segment for the beam
    ramp_length = segment_length *