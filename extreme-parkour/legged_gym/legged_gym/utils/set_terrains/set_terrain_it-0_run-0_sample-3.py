import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of staggered, sloped ramps for the robot to walk up, turn and walk down, testing slope traversal and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((4, 2))

    # Ramp/landing dimensions
    ramp_length = 2.2 - 0.8 * difficulty        # ramps are shorter and steeper with difficulty
    ramp_length_idx = m_to_idx(ramp_length)
    ramp_width = 1.2
    ramp_width_idx = m_to_idx(ramp_width)
    max_ramp_height = 0.09 + 0.35 * difficulty  # ramps are higher with harder difficulty
    slope_up = max_ramp_height / ramp_length
    landing_length = 0.6                        # flat between ramps/at top
    landing_length_idx = m_to_idx(landing_length)
    landing_height = max_ramp_height

    # Flat spawn area
    spawn_length = m_to_idx(2)
    mid_y = m_to_idx(width) // 2
    height_field[0:spawn_length, :] = 0

    # The overall plan:
    # - ramp up (centered), flat, turn left, ramp down (left), flat, ramp up (right), flat, ramp down (exit), flat exit
    # - Goals at key points: bottom of first ramp, top turning flat, top of third ramp, exit flat

    # RAMP 1: Upward, centered
    ramp1_start_x = spawn_length
    ramp1_end_x = ramp1_start_x + ramp_length_idx
    ramp1_center_y = mid_y
    ramp1_y1 = ramp1_center_y - ramp_width_idx // 2
    ramp1_y2 = ramp1_center_y + ramp_width_idx // 2

    # Linearly increase the height for ramp up
    for i in range(ramp1_length := ramp1_end_x - ramp1_start_x):
        height = (i / max(1, ramp1_length-1)) * max_ramp_height
        height_field[ramp1_start_x + i, ramp1_y1:ramp1_y2] = height

    # LANDING/TURN PLATFORM 1 (at top, left side)
    turn_landing1_x_start = ramp1_end_x
    turn_landing1_x_end = turn_landing1_x_start + landing_length_idx
    turn1_y_center = ramp1_center_y - m_to_idx(0.8)
    turn1_y1 = turn1_y_center - ramp_width_idx // 2
    turn1_y2 = turn1_y_center + ramp_width_idx // 2
    height_field[turn_landing1_x_start:turn_landing1_x_end, turn1_y1:turn1_y2] = landing_height

    # RAMP 2: Downward, left side (left lane)
    ramp2_start_x = turn_landing1_x_end
    ramp2_end_x = ramp2_start_x + ramp_length_idx
    ramp2_y1 = turn1_y1
    ramp2_y2 = turn1_y2

    for i in range(ramp2_length := ramp2_end_x - ramp2_start_x):
        height = landing_height - (i / max(1, ramp2_length-1)) * max_ramp_height
        height_field[ramp2_start_x + i, ramp2_y1:ramp2_y2] = height

    # LANDING/TURN PLATFORM 2 (bottom, right side)
    turn_landing2_x_start = ramp2_end_x
    turn_landing2_x_end = turn_landing2_x_start + landing_length_idx
    turn2_y_center = ramp2_y1 + m_to_idx(1.4)
    turn2_y1 = turn2_y_center - ramp_width_idx // 2
    turn2_y2 = turn2_y_center + ramp_width_idx // 2
    height_field[turn_landing2_x_start:turn_landing2_x_end, turn2_y1:turn2_y2] = 0

    # RAMP 3: Upward again, right lane
    ramp3_start_x = turn_landing2_x_end
    ramp3_end_x = ramp3_start_x + ramp_length_idx
    ramp3_y1 = turn2_y1
    ramp3_y2 = turn2_y2

    for i in range(ramp3_length := ramp3_end_x - ramp3_start_x):
        height = (i / max(1, ramp3_length-1)) * max_ramp_height
        height_field[ramp3_start_x + i, ramp3_y1:ramp3_y2] = height

    # LANDING/TURN PLATFORM 3 (top, near center again)
    turn_landing3_x_start = ramp3_end_x
    turn_landing3_x_end = turn_landing3_x_start + landing_length_idx
    turn3_y_center = turn2_y_center - m_to_idx(1.2)
    turn3_y1 = turn3_y_center - ramp_width_idx // 2
    turn3_y2 = turn3_y_center + ramp_width_idx // 2
    height_field[turn_landing3_x_start:turn_landing3_x_end, turn3_y1:turn3_y2] = landing_height

    # RAMP 4: Down exit, center/right
    ramp4_start_x = turn_landing3_x_end
    ramp4_end_x = ramp4_start_x + ramp_length_idx
    ramp4_y1 = turn3_y1
    ramp4_y2 = turn3_y2

    for i in range(ramp4_length := ramp4_end_x - ramp4_start_x):
        height = landing_height - (i / max(1, ramp4_length-1)) * max_ramp_height
        height_field[ramp4_start_x + i, ramp4_y1:ramp4_y2] = height

    # EXIT FLAT
    height_field[min(ramp4_end_x, height_field.shape[0]):, :] = 0

    # Set the goals: (start, turn top, second top, end)
    goals[0] = [ramp1_start_x + m_to_idx(0.3), ramp1_center_y]                 # Entry at base of first ramp
    goals[1] = [turn_landing1_x_start + landing_length_idx//2, turn1_y_center] # Top of first climb (left turn)
    goals[2] = [turn_landing3_x_start + landing_length_idx//2, turn3_y_center] # Top of last climb (right turn)
    goals[3] = [min(ramp4_end_x + m_to_idx(0.5), height_field.shape[0]-1), turn3_y_center]  # End after final ramp

    return height_field, goals