import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of staggered 'stepping stones' (wide, flat pillars) over a trench to test accurate foot placement and agile navigation."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        if isinstance(m, (list, tuple)):
            return [round(i / field_resolution) for i in m]
        return np.round(m / field_resolution).astype(np.int16)

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Terrain concept:
    # Starting at x=2m, the robot must cross a long trench using a sequence of
    # circular or oval "stepping stone" platforms, with gaps between them.
    # The stones are always above center ground height (flat at 0), the gaps are deep dips.
    # Requires the quadruped to be accurate and agile with paw placement,
    # leveraging vision and whole-body agility, with no climbing required.
    # Stones are staggered with lateral (y) offsets to force turning and path planning.

    trench_depth = -(0.25 + 0.9 * difficulty)  # Pit is deeper with higher difficulty, up to -1.15m
    stone_height = 0.10 + 0.20 * difficulty    #