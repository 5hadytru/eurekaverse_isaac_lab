import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of ascending stairs and step-downs, challenging the robot's step-height planning and stability."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # Parameters
    # Each step is at least as wide as the quadruped, spanning the full course width
    stair_width = m_to_idx(width)
    # Step depth: how long (in x) is each step? More difficult = smaller steps
    step_depth = 0.7 - 0.4 * difficulty    # m, min 0.3m
    step_depth_idx = m_to_idx(step_depth)
    # Step height: how tall is each step? More difficult = taller steps
    max_step_height = 0.17 + 0.13 * difficulty   # 0.17 (gentle) to 0.30 (challenging)
    min_step_height = 0.11 + 0.06 * difficulty   # 0.11 (low) to 0.17 (tough)
    # Number of step ups before a step down
    num_ascend_steps = 2
    num_descend_steps = 2
    num_sets = 2      # How many up-down sets (total 4 steps up, 4 steps down ~ 8 steps)

    # Ensure we leave spawn area and final area flat
    spawn_length = m_to_idx(2.0)
    end_pad = m_to_idx(2.0)
    mid_y = m_to_idx(width/2)

    # Set spawn flat
    height_field[0:spawn_length, :] = 0
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    cur_x = spawn_length
    cur_height = 0

    goal_idx = 1

    # MAIN: alternating up-steps and down-steps
    for set_num in range(num_sets):
        # Ascend
        for i in range(num_ascend_steps):
            step_h = np.random.uniform(min_step_height, max_step_height)
            # Build a riser
            next_x = cur_x + step_depth_idx
            height_field[cur_x:next_x, :] = cur_height + step_h
            cur_height += step_h

            # Place goal at second ascent and second descent
            if ((goal_idx < 5) and ((set_num == 0 and i == 0) or (set_num == 0 and i == 1) or (set_num == 1 and i == 0) or (set_num == 1 and i == 1))):
                goals[goal_idx] = [cur_x + step_depth_idx // 2, mid_y]
                goal_idx += 1

            cur_x = next_x
        # Descend
        for i in range(num_descend_steps):
            step_h = np.random.uniform(min_step_height, max_step_height)
            next_x = cur_x + step_depth_idx
            # Step down, i.e., negative
            height_field[cur_x:next_x, :] = cur_height - step_h
            cur_height -= step_h
            cur_x = next_x

    # Ensure no abrupt cliff at the end
    height_field[cur_x:, :] = 0
    if goal_idx < 5:
        goals[goal_idx] = [cur_x + m_to_idx(0.5), mid_y]
        goal_idx += 1

    # Fill any remaining goals at the very end
    while goal_idx < 5:
        goals[goal_idx] = [m_to_idx(length) - m_to_idx(0.5), mid_y]
        goal_idx += 1

    return height_field, goals