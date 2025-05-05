import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping-stone sequence: sequence of spaced narrow flat stepping pads over a trench, forcing controlled foot placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- Terrain Parameters ---
    # Length and width of each stone (narrow - only just larger than robot's footprint)
    stone_width = 0.45 + 0.15 * (1.0 - difficulty)          # 0.45–0.60m, easier=larger
    stone_length = 0.5 + 0.25 * (1.0 - difficulty)          # 0.5–0.75m

    # Lateral arrangement ("wobble"): stones alternate left, right of nominal center
    max_y_offset = 0.55 * difficulty                        # Pad centers move off midline up to ±0.55m (hard)

    # Vertical arrangement: stones are 0.10–0.22m above the pit,