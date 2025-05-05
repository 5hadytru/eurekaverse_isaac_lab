import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Sequence of ascending and descending ramps (A-frames) for the quadruped to walk over in parkour style."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course idea: 
    # A series of 5 serious A-frame ramp obstacles (up, over, and down), like those seen in dog agility courses, 
    # spanning the width of the field. The ramps force the robot to climb up, balance on the crest, 
    # and descend, challenging balance, grip, and coordination.
    # 
    # On higher difficulty, ramps get taller, steeper, and closer together.
    # On lower difficulty, they are flatter and more spaced out.

    # Ramp settings
    n_ramps = 5
    ramp_length = 1.6 - 0.4 * difficulty            # meters, base length of one side of A
    ramp_height = 0.12 + 0.28 * difficulty          #