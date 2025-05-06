import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of sloped ramps and stairs for testing quadruped's dynamic balance up/down inclines."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course parameters
    ramp_count = 4  # Number of sloped ramps in series
    ramp_length = 1.8 - 0.6 * difficulty  # ramps get shorter (steeper) as difficulty increases
    ramp_length_idx = m_to_idx(ramp_length)
    ramp_width = 1.4
    ramp_width_idx = m_to_idx(ramp_width)
    max_ramp_height = 0.32 + 0.22 * difficulty  # ~0.32m to 0.54m at max difficulty
    stair_rise = 0.08 + 0.06 * difficulty       # height of each stair
    stair_count = 2 + int(3 * difficulty)       # number of stairs
    stair_section_length = 0.7 + 0.6 * difficulty
    stair_section_length_idx = m_to_idx(stair_section_length)
    stair_width = 1.4
    stair_width_idx = m_to_idx(stair_width)
    pit_length = 0.35 + 0.15 * difficulty

    mid_y = m_to_idx(width/2)
    # Flat spawn area
    spawn_length = m_to_idx(2.0)
    height_field[0:spawn_length, :] = 0
    goals[0] = [m_to_idx(1.0), mid_y]

    cur_x = spawn_length

    # Helper: add centered ramp
    def add_ramp(x_start, x_end, y_c, w_idx, h0, h1):
        y1 = y_c - w_idx//2
        y2 = y_c + w_idx//2
        for xi in range(x_start, x_end):
            phase = (xi - x_start) / (x_end - x_start)
            height = h0 + (h1-h0)*phase
            height_field[xi, y1:y2] = height

    # Helper: add centered stairs
    def add_stairs(x_start, x_end, y_c, w_idx, h0, rise, stair_num):
        y1 = y_c - w_idx//2
        y2 = y_c + w_idx//2
        sec_len = (x_end - x_start) // stair_num
        for si in range(stair_num):
            sx1 = x_start + si*sec_len
            sx2 = min(x_start + (si+1)*sec_len, x_end)
            h = h0 + rise*si
            height_field[sx1:sx2, y1:y2] = h

    # Obstacles and goals loop
    # Sequence: Ramp up -> down stairs -> pit -> Ramp down -> up stairs -> repeat etc.
    for i in range(ramp_count):
        # Ramp up
        ramp_height = max_ramp_height * (0.85 + 0.3*random.random())
        ramp_dir = 1 if (i%2==0) else -1  # up/down alternation
        next_height = (height_field[cur_x-1, mid_y] if cur_x>0 else 0) + ramp_dir*ramp_height

        add_ramp(cur_x, cur_x + ramp_length_idx, mid_y, ramp_width_idx,
                 height_field[cur_x-1, mid_y] if cur_x>0 else 0, next_height)
        # Place a goal at center ramp
        goals[2*i+1] = [cur_x + ramp_length_idx//2, mid_y]
        cur_x += ramp_length_idx

        # Down/Up stairs after ramp
        stair_h0 = height_field[cur_x-1, mid_y]
        stair_h1 = stair_h0 - ramp_dir*stair_rise*stair_count  # Ascend or descend stairs depending on ramp
        add_stairs(cur_x, cur_x+stair_section_length_idx, mid_y, stair_width_idx, stair_h0, -ramp_dir*stair_rise, stair_count)
        goals[2*i+2] = [cur_x + stair_section_length_idx//2, mid_y]
        cur_x += stair_section_length_idx

        # Pit or flat section between obstacles. At low difficulty, pit is shallow.
        pit_depth = -0.3 - 0.2*difficulty   # force the robot to stay on obstacles
        pit_start = cur_x
        pit_end = cur_x + m_to_idx(pit_length)
        height_field[pit_start:pit_end, :] = pit_depth
        cur_x = pit_end

    # Finish section: flat terrain
    remain = height_field.shape[0] - cur_x
    if remain > 0:
        height_field[cur_x:, :] = 0
        goals[7] = [height_field.shape[0]-m_to_idx(1.0), mid_y]
    else:
        goals[7] = [height_field.shape[0]-m_to_idx(1.0), mid_y]

    # Check that all goals are inside the field boundaries and adjust if needed
    for gi in range(goals.shape[0]):
        x, y = goals[gi]
        x = min(max(0, int(x)), height_field.shape[0]-1)
        y = min(max(0, int(y)), height_field.shape[1]-1)
        goals[gi] = [x, y]

    return height_field, goals