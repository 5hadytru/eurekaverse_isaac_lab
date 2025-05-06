import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Alternating ramps and staircases traverse: tests climbing, descending, and traversing stairs."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # The robot size
    robot_length = 0.645
    robot_width = 0.28

    # Key placement variables
    mid_y = m_to_idx(width) // 2
    spawn_length = m_to_idx(2.0)
    # Start: always flat
    height_field[:spawn_length, :] = 0.0
    goals[0] = [m_to_idx(1.0), mid_y]  # Initial goal at the center of flat spawn

    # Obstacle parameters (all relative to difficulty)
    min_ramp_height = 0.08 + 0.10 * difficulty   # Steeper at higher difficulty
    max_ramp_height = 0.12 + 0.18 * difficulty
    ramp_length_min = 1.5 - 0.2 * difficulty
    ramp_length_max = 2.1 - 0.4 * difficulty
    ramp_width = 1.2 + 0.4 * (1-difficulty)      # Wide at low difficulty
    # Stairs
    num_stairs_min = 3
    num_stairs_max = 6 + int(difficulty*2)
    stair_width = 1.2
    stair_length = 1.2 - 0.4 * difficulty
    stair_height_min = 0.06 + 0.07 * difficulty
    stair_height_max = 0.12 + 0.09 * difficulty

    # Obstacles sequence (ramp up, stairs up, ramp down, stairs down, repeat)
    cur_x = spawn_length
    step = 1  # index in the obstacles/goals
    for obstacle_num in range(3):
        # --- RAMP UP ---
        ramp_length = random.uniform(ramp_length_min, ramp_length_max)
        ramp_length_idx = m_to_idx(ramp_length)
        ramp_width_idx = m_to_idx(ramp_width)
        ramp_start = cur_x
        ramp_end = ramp_start + ramp_length_idx
        ramp_y1 = mid_y - ramp_width_idx//2
        ramp_y2 = mid_y + ramp_width_idx//2
        ramp_h = random.uniform(min_ramp_height, max_ramp_height)

        # Make ramp height linearly increasing
        height_field[ramp_start:ramp_end, ramp_y1:ramp_y2] = np.linspace(
            0, ramp_h, ramp_end - ramp_start)[:, None]
        cur_x = ramp_end
        goals[step] = [ramp_start + (ramp_length_idx // 2), mid_y]
        step += 1

        # --- STAIRS UP ---
        num_stairs = random.randint(num_stairs_min, num_stairs_max)
        stair_tread_length = m_to_idx(stair_length / num_stairs)
        stair_rise = np.linspace(0, random.uniform(stair_height_min, stair_height_max), num_stairs+1)
        stairs_start = cur_x
        stairs_y1 = mid_y - m_to_idx(stair_width//2)
        stairs_y2 = mid_y + m_to_idx(stair_width//2)
        for s in range(num_stairs):
            height_field[
                stairs_start + s*stair_tread_length:
                stairs_start + (s+1)*stair_tread_length, 
                stairs_y1:stairs_y2] = stair_rise[s+1]
        cur_x = stairs_start + num_stairs*stair_tread_length
        goals[step] = [stairs_start + (num_stairs*stair_tread_length)//2, mid_y]
        step += 1

        # --- RAMP DOWN ---
        ramp_length = random.uniform(ramp_length_min, ramp_length_max)
        ramp_length_idx = m_to_idx(ramp_length)
        ramp_end2 = cur_x + ramp_length_idx
        ramp_h2 = 0.0  # back to floor level
        prev_h = height_field[cur_x-1, mid_y]  # current height
        height_field[cur_x:ramp_end2, ramp_y1:ramp_y2] = np.linspace(
            prev_h, ramp_h2, ramp_end2 - cur_x)[:, None]
        goals[step] = [cur_x + (ramp_length_idx // 2), mid_y]
        cur_x = ramp_end2
        step += 1

        # --- STAIRS DOWN ---
        num_stairs = random.randint(num_stairs_min, num_stairs_max)
        stair_tread_length = m_to_idx(stair_length / num_stairs)
        stair_rise = np.linspace(prev_h, 0, num_stairs+1)
        stairs_start = cur_x
        for s in range(num_stairs):
            height_field[
                stairs_start + s*stair_tread_length:
                stairs_start + (s+1)*stair_tread_length, 
                stairs_y1:stairs_y2] = stair_rise[s+1]
        cur_x = stairs_start + num_stairs*stair_tread_length
        goals[step] = [stairs_start + (num_stairs*stair_tread_length)//2, mid_y]
        step += 1

    # Fill rest with flat ground if needed
    height_field[cur_x:, :] = 0.0
    if step < 8:
        # Place trailing goals to flat
        for i in range(step, 8):
            goals[i] = [cur_x + m_to_idx(0.4*(i-step+1)), mid_y]

    # Clip all goal indices to within field bounds
    goals = np.clip(goals, [0,0],[height_field.shape[0]-1, height_field.shape[1]-1])

    return height_field, goals