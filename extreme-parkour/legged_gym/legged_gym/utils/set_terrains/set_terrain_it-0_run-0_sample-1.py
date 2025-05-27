import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of angled 'A-frame ramps' and low hurdles that test the quadruped's climbing and descending on sloped surfaces."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Spawn and flat entry area
    spawn_length = m_to_idx(2)
    height_field[:spawn_length, :] = 0

    # Ramp/hurdle setup
    num_obstacles = 6
    space_length = length - 2.2  # 2m spawn, 0.2m buffer at end
    obs_spacing = space_length / num_obstacles
    ramp_base = 1.2 - 0.4 * difficulty    # Ramp "floor" width: 1.2m to 0.8m
    ramp_base = max(0.8, ramp_base)
    ramp_height = 0.15 + 0.25 * difficulty  # 0.15m to 0.40m at hardest
    hurdle_height = 0.08 + 0.10 * difficulty  # Small, can step over

    mid_y = m_to_idx(width / 2)
    w = m_to_idx(width)
    ramp_width = m_to_idx(1.2)  # Wide enough so robot can swerve slightly

    # Stagger: alternate: ramp, hurdle, ramp, hurdle, ramp, hurdle
    obs_list = []
    for i in range(num_obstacles):
        x_start = spawn_length + int(i * obs_spacing)
        x_end = x_start + m_to_idx(ramp_base)
        y1 = mid_y - ramp_width//2
        y2 = y1 + ramp_width

        if i % 2 == 0:
            # Create an A-frame ramp (ascending then descending in succession)
            half_ramp = (x_end - x_start)//2
            # Ascend
            for j in range(half_ramp):
                height_field[x_start + j, y1:y2] = (j / (half_ramp)) * ramp_height
            # Descend
            for j in range(half_ramp, x_end - x_start):
                height_field[x_start + j, y1:y2] = ramp_height - ((j - half_ramp)/(half_ramp)) * ramp_height
            # Clamp at edges
            height_field[x_start:x_end, 0:y1] = 0
            height_field[x_start:x_end, y2:] = 0
            # Mid-ramp goal at the top
            goals[i+1] = [x_start + half_ramp, mid_y]
            obs_list.append(('ramp', x_start, x_end, y1, y2))
        else:
            # Hurdle is a low, blocky obstacle stretching across the course
            hurdle_length = m_to_idx(0.45 + 0.15 * difficulty)
            x1 = x_start
            x2 = min(x_start + hurdle_length, m_to_idx(length)-1)
            height_field[x1:x2, :] = hurdle_height
            # Place goal just after hurdle
            goals[i+1] = [int((x1 + x2) / 2), mid_y]
            obs_list.append(('hurdle', x1, x2, 0, w))

    # Final goal at the end on flat terrain
    goals[0] = [m_to_idx(1), mid_y]   # start
    goals[7] = [m_to_idx(length)-m_to_idx(0.5), mid_y]

    # Buffer zone at the end for the robot to stop safely
    height_field[-m_to_idx(0.2):, :] = 0

    return height_field, goals