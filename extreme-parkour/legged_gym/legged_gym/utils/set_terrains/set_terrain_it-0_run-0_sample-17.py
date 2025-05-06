import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of sloped ramps (uphill and downhill) alternating direction, testing the quadruped's ability to ascend, descend, and change heading while maintaining balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))
    
    # COURSE CONFIGURATION ---------------------------
    # We will alternate sloped ramps going up and down
    # Each ramp will span ~1.4m to 1.8m length, width 1.3 to 1.8m
    # Slope angle and ramp height scale with difficulty

    min_ramp_len, max_ramp_len = 1.4, 1.8
    min_ramp_wid, max_ramp_wid = 1.3, 1.8
    max_total_height = 0.20 + 0.35 * difficulty  # Steeper slope at higher difficulty

    # We'll use 6 ramps, each with an up or down depending on position, alternating direction
    # Space between ramps is flat
    n_ramps = 6
    flat_space_len = 0.40 + 0.12 * (1 - difficulty) # shorter rest space at higher difficulty

    # The course will zig-zag: first ramp up, then turn left, ramp down, then turn right, etc.
    # We'll select y positions to zig-zag across the width
    width_margin = 0.45  # leave some gap on sides for safe navigation
    ramp_centers_y = np.linspace(width_margin, width - width_margin, n_ramps + 1)
    # Random swap every other direction
    if random.random() > 0.5:
        ramp_centers_y = ramp_centers_y[::-1]

    # Place the spawn and first goal at flat area at (1m, center)
    spawn_x = 1.0
    spawn_x_idx = m_to_idx(spawn_x)
    goals[0] = [spawn_x_idx, m_to_idx(width/2)]

    # Set spawn area to flat
    height_field[:m_to_idx(2.0), :] = 0.0

    cur_x = 2.0  # start placing obstacles after safe spawn

    ramp_goals_idx = 1
    for ramp_i in range(n_ramps):
        # Ramp parameters
        ramp_length_m = np.random.uniform(min_ramp_len, max_ramp_len)
        ramp_width_m = np.random.uniform(min_ramp_wid, max_ramp_wid)
        ramp_height = (max_total_height * ((-1) ** ramp_i))  # alternate up and down ramps
        ramp_start_x = cur_x
        ramp_end_x = cur_x + ramp_length_m
        ramp_mid_y = ramp_centers_y[ramp_i % (n_ramps + 1)]

        start_y = np.clip(m_to_idx(ramp_mid_y - ramp_width_m/2), 0, m_to_idx(width)-1)
        end_y = np.clip(m_to_idx(ramp_mid_y + ramp_width_m/2), 1, m_to_idx(width))
        y_slice = slice(start_y, end_y)
        x_start_idx = m_to_idx(ramp_start_x)
        x_end_idx = m_to_idx(ramp_end_x)
        x_slice = slice(x_start_idx, x_end_idx)

        # Compute slope for every x across the ramp
        ramp_len_idx = x_end_idx - x_start_idx
        ramp_slope = np.linspace(0, ramp_height, ramp_len_idx).reshape(-1, 1)
        # Broadcast slope over width
        height_field[x_slice, y_slice] = height_field[x_slice, y_slice] + ramp_slope

        # Set sides flat to force following the ramp
        if start_y > 0:
            height_field[x_slice, :start_y] = -0.15  # 15cm pit
        if end_y < m_to_idx(width):
            height_field[x_slice, end_y:] = -0.15

        # Intermediate flat section after each ramp for stability and goal
        flat_length_m = flat_space_len
        flat_start_x = ramp_end_x
        flat_end_x = ramp_end_x + flat_length_m
        x_flat_start_idx = m_to_idx(flat_start_x)
        x_flat_end_idx = m_to_idx(flat_end_x)
        # Maintain ending height of ramp throughout the flat
        landing_height = ramp_height
        if ramp_i == 0:
            # Ramp height is relative to previous ground, which may not be zero
            base_height = 0.0
        else:
            base_height = height_field[m_to_idx(cur_x-0.01), m_to_idx(ramp_mid_y)]
        base_height = height_field[x_start_idx, m_to_idx(ramp_mid_y)]
        height_field[x_flat_start_idx:x_flat_end_idx, y_slice] = base_height + ramp_height

        # Put a goal at the end of the ramp
        y_goal = np.clip(m_to_idx(ramp_mid_y), 0, m_to_idx(width)-1)
        x_goal = int((x_end_idx + x_flat_start_idx)//2)
        goals[ramp_goals_idx] = [x_goal, y_goal]
        ramp_goals_idx += 1

        cur_x = flat_end_x

    # Pad the rest of the field flat
    height_field[m_to_idx(cur_x):, :] = height_field[m_to_idx(cur_x)-1, :][np.newaxis, :]

    # If less than 8 goals, place final ones at the end, spaced out
    last_goal_x = m_to_idx(cur_x) + m_to_idx(0.2)
    for i in range(ramp_goals_idx, 8):
        goals[i] = [min(last_goal_x + m_to_idx(0.3*(i - ramp_goals_idx)), m_to_idx(length)-2), m_to_idx(width/2)]

    return height_field, goals