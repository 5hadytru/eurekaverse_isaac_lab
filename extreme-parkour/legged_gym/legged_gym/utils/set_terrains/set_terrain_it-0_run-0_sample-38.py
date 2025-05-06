import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of angled ramps to test dynamic balance when climbing/descending and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for ramp obstacles
    min_ramp_length = 1.2
    max_ramp_length = 2.5
    min_ramp_width = 1.2
    max_ramp_width = 2.0
    min_angle = 8    # degrees
    max_angle = 26   # degrees
    padding = 0.4    # space between ramps (meters)
    shift_y = 0.8    # how far off-center ramps can be
    num_ramps = 4

    # Ramp size and angles scale with difficulty
    ramp_lengths = np.linspace(min_ramp_length, max_ramp_length, num_ramps) + difficulty * 0.3
    ramp_widths = np.linspace(max_ramp_width, min_ramp_width, num_ramps)
    ramp_angles = np.linspace(min_angle, max_angle, num_ramps) + difficulty * 10.0
    ramp_angles = np.clip(ramp_angles, min_angle, max_angle)

    spawn_length = m_to_idx(2)
    mid_y = m_to_idx(width / 2)

    # Ensure start region is clear and the first goal is just after spawn
    height_field[0:spawn_length, :] = 0
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    # Create angled ramps alternatively left and right, with elevation changes, forming an S-bend course
    cur_x = spawn_length
    sgn = 1  # Used to alternate ramp direction (left/right)
    for i in range(num_ramps):
        ramp_len = m_to_idx(ramp_lengths[i])
        ramp_wid = m_to_idx(ramp_widths[i])
        ramp_ang_rad = np.deg2rad(ramp_angles[i])

        # Y-position offset to cause zig-zag; padding ensures ramps stay within field
        y_offset = int(sgn * m_to_idx(shift_y))
        y_center = np.clip(mid_y + y_offset, m_to_idx(0.5 * ramp_wid), m_to_idx(width - 0.5 * ramp_wid) - 1)

        x1 = cur_x
        x2 = np.clip(x1 + ramp_len, 0, m_to_idx(length)-1)
        y1 = max(y_center - ramp_wid // 2, 0)
        y2 = min(y_center + ramp_wid // 2, m_to_idx(width)-1)

        # Height delta for this ramp, positive up, negative ramps for descending at high difficulty
        h_sign = 1 if i % 2 == 0 else (-1 if difficulty > 0.4 else 1)
        h_delta = h_sign * ramp_len * field_resolution * np.tan(ramp_ang_rad)

        # Ramp heights: start at the previous ramp's elevation
        if i == 0:
            base_h = 0
        else:
            base_h = float(height_field[x1-1, mid_y])

        for xi in range(x1, x2):
            frac = (xi - x1) / max(1, (x2 - x1 - 1))
            ramp_h = base_h + h_delta * frac
            height_field[xi, y1:y2] = ramp_h

        # Place a goal at the center, portion way up the ramp (not exactly at the tip to avoid edges)
        goal_x = int(x1 + 0.7 * (x2 - x1))
        goal_y = int((y1 + y2)/2)
        goals[i+1] = [goal_x, goal_y]

        # Add flat ground/padding between ramps, ensure transition is not abrupt
        pad_len = m_to_idx(padding + 0.2 * difficulty)
        next_base = base_h + h_delta
        x_pad_start = x2
        x_pad_end = min(x_pad_start + pad_len, m_to_idx(length)-1)
        if x_pad_start < x_pad_end:
            height_field[x_pad_start:x_pad_end, :] = next_base
        cur_x = x_pad_end

        # Change ramp direction
        sgn *= -1

    # Place further flat ground at the end, and final goals
    end_flat = m_to_idx(2.0)
    height_field[cur_x:cur_x+end_flat, :] = float(height_field[cur_x-1, mid_y])
    goals[5] = [cur_x + end_flat // 3, mid_y]
    goals[6] = [cur_x + 2*end_flat // 3, mid_y]
    # Last goal at course end
    goals[7] = [m_to_idx(length) - m_to_idx(0.6), mid_y]

    return height_field, goals