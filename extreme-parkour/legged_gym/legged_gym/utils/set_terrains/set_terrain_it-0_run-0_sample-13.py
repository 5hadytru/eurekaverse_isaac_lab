import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of staggered 'stepping-stone' beams crossing over a trench, testing precise foot placement and straight walking."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Basic geometry
    spawn_length = m_to_idx(2.0)
    terrain_length = m_to_idx(length)
    terrain_width = m_to_idx(width)
    mid_y = terrain_width // 2

    # Parameters for beams and trench
    # Beam width: between 0.45 and 0.7m (challenge: requires foot placement, but always > leg stance)
    beam_width = np.interp(difficulty, [0, 1], [0.7, 0.45])
    beam_width_idx = m_to_idx(beam_width)
    # Beam length: spans almost the terrain width
    beam_length = width - 0.15
    beam_length_idx = m_to_idx(beam_length)
    # Height of the beam above ground