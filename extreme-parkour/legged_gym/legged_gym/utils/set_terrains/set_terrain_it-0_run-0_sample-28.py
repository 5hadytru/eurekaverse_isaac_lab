import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Slalom-style zig-zag wall course testing precise lateral maneuvering and tight turns."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        if isinstance(m, list) or isinstance(m, tuple):
            return [round(i / field_resolution) for i in m]
        return np.round(m / field_resolution).astype(np.int16)

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Wall dimensions: 1m wide, adjustable thickness, gaps just wider than quadruped
    wall_length = 1.0
    wall_thickness = 0.15 + 0.15 * difficulty  # walls get thicker with difficulty
    wall_gap = 0.35 + 0.50 * (1 - difficulty)  # gap narrows with difficulty, min about 0.35m (narrow for the bot!)

    wall_height = 0.25 + 0.35 * difficulty  # must be non-traversable height, taller when harder

    n_walls = 6
    start_buffer_x = 2.0  # No obstacles where the bot spawns
    end_buffer_x = 1.0    # Leave space before the finish line

    #