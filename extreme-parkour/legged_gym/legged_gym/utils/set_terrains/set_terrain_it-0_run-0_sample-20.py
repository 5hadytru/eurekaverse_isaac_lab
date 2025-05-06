import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of staggered low-walls ('hurdles') for the quadruped to step over or jump, testing dynamic gait and body lifting."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # ------------- Obstacle course config -------------

    # Basic layout: sequence of 6 to 7 staggered low hurdles with alternating lateral positions to force slight path adjustment.

    # Hurdle parameters:
    hurdle_length_m = 0.15 + 0.15 * difficulty           # Make hurdle just thick enough to track collisions
    hurdle_width_m  = 1.2 + 0.8 * (1-difficulty)         # At least 1.2m, narrower for high diff but never too thin
    hurdle_height_min = 0.06 + 0.07 * difficulty         # Min height at easy, up to ~0.13 at hard 
    hurdle_height_max = 0.10 + 0.2 * difficulty          # Up to 0.3m for max
    inter_hurdle_x = 1.2 - 0.25 * difficulty             # Space between hurdles
    lateral_offset = 0.4 + 0.25 * difficulty             # How off center hurdles are
    spawn_length = m_to_idx(2.0)

    # Compute hurdles:
    num_hurdles = 6 + (1 if difficulty > 0.7 else 0)    # Add an extra hurdle when difficulty is high

    # Place spawn area flat
    height_field[:spawn_length, :] = 0

    mid_y = m_to_idx(width / 2)
    width_n = m_to_idx(width)

    # Generate hurdles
    cur_x = spawn_length
    lateral_sign = 1
    for i in range(num_hurdles):
        hurdle_len = m_to_idx(hurdle_length_m)
        hurdle_wid = m_to_idx(hurdle_width_m)
        # Stagger hurdle position in y, but keep within bounds
        offset_y = int(lateral_sign * (m_to_idx(lateral_offset)))
        hurdle_center_y = np.clip(mid_y + offset_y, m_to_idx(0.5), width_n - m_to_idx(0.5))
        y1 = max(0, hurdle_center_y - hurdle_wid // 2)
        y2 = min(width_n, hurdle_center_y + hurdle_wid // 2)
        x1 = cur_x
        x2 = min(m_to_idx(length), cur_x + hurdle_len)
        hurdle_height = np.random.uniform(hurdle_height_min, hurdle_height_max)
        height_field[x1:x2, y1:y2] = hurdle_height

        # Insert a goal right after each hurdle, so the robot must pass each one directly
        # Place goal just after the hurdle, slightly further along x, and centered with the hurdle
        goals[i+1] = [x2 + m_to_idx(0.2), hurdle_center_y]

        # Stagger next hurdle:
        lateral_sign *= -1
        # Set inter-hurdle distance with jitter for realism
        cur_x = x2 + m_to_idx(inter_hurdle_x + np.random.uniform(-0.12, 0.15))

    # Place first goal at spawn area, center (slightly before first hurdle)
    goals[0] = [m_to_idx(1.25), mid_y]

    # Final goal at the end area, centered
    goals[7] = [m_to_idx(length) - m_to_idx(0.7), mid_y]

    # Fill between hurdles and after last hurdle with flat ground
    height_field[cur_x:, :] = 0

    return height_field, goals