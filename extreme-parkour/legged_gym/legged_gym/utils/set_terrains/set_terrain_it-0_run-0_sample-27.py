import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of sloped ramp obstacles of varying steepness, forming a zig-zag course to test climbing, stability, and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # General parameters
    course_length_idx, course_width_idx = m_to_idx(length), m_to_idx(width)
    spawn_x = m_to_idx(1)
    spawn_y = course_width_idx // 2

    # Obstacle parameters (ramps)
    #
    # The course will be a sequence of angled ramps with flat sections between them,
    # and the robot will alternately turn left and right up each ramp segment.
    #
    ramp_length_m = 1.6 - 0.5 * difficulty      # Each ramp segment, meters
    ramp_length = m_to_idx(ramp_length_m)
    ramp_width_m = 1.1
    ramp_width = m_to_idx(ramp_width_m)
    flat_section_length_m = 0.5 - 0.2 * difficulty  # Flat between ramps, meters
    flat_section_length = m_to_idx(flat_section_length_m)
    max_ramp_height = 0.29 + 0.22 * difficulty        # meters per ramp

    # The robot starts on flat ground
    height_field[:spawn_x+1, :] = 0
    goals[0] = [spawn_x, spawn_y]   # spawn goal

    # Slope sequence setup
    # The robot will have to zig-zag: alternate left and right, with angled ramps
    num_ramps = 6
    direction = 1    # 1 = up left, -1 = up right
    cur_x = spawn_x + 1
    cur_y = spawn_y
    goals_filled = 1

    for i in range(num_ramps):
        # Calculate the start and end points for this ramp
        # Offset sideways (y direction) for zig-zag
        # At high difficulty, ramps are shifted wider
        y_offset = int((course_width_idx // 2.8) * (0.6 + 0.7 * difficulty) * direction)

        ramp_start_x = cur_x
        ramp_end_x = cur_x + ramp_length

        # Clamp ramp position
        ramp_start_y = np.clip(cur_y - ramp_width//2 + y_offset, m_to_idx(0.5), course_width_idx - ramp_width - m_to_idx(0.5))
        ramp_start_y = int(ramp_start_y)
        ramp_end_y = ramp_start_y + ramp_width

        # Make the ramp: linearly increasing height along x
        for x in range(ramp_start_x, min(ramp_end_x, course_length_idx)):
            frac_on_ramp = (x - ramp_start_x) / max((ramp_end_x - ramp_start_x), 1)
            height = frac_on_ramp * max_ramp_height
            height_field[x, ramp_start_y:ramp_end_y] = height

        # Place a goal at the center top of the ramp
        goal_x = int(ramp_end_x - ramp_length // 2)
        goal_y = int(ramp_start_y + ramp_width//2)
        if goals_filled < 8:
            goals[goals_filled] = [goal_x, goal_y]
            goals_filled += 1

        # Add a flat section after the ramp at same ramp top height
        flat_start_x = ramp_end_x
        flat_end_x = flat_start_x + flat_section_length
        height_field[flat_start_x:flat_end_x, ramp_start_y:ramp_end_y] = max_ramp_height

        # Set next start position for following ramp (keep on the flat)
        cur_x = int(flat_end_x)
        cur_y = goal_y

        # Alternate directions to zig-zag
        direction *= -1

    # Final flat section to goal (goal 7)
    flat_final_len = m_to_idx(1.0)
    final_x = min(cur_x + flat_final_len, course_length_idx - 1)
    final_y = cur_y
    height_field[cur_x:final_x, :] = max_ramp_height

    # Place final goals at the far end of the course
    for i in range(goals_filled, 8):
        goals[i] = [final_x - 1, final_y]

    return height_field, goals