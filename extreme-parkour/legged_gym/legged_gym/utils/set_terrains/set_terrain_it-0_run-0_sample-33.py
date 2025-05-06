import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Slalom course with tall barriers for the quadruped to weave around and squeeze through."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # SLALOM DETAILS
    # Tall barriers for the dog to weave through. 
    # Encourages tight turning, steering, and side-stepping, with barrier widths/narrowness set by difficulty.

    # Constants for obstacles
    num_barriers = 6
    min_gap = 0.5         # Minimum gap between barrier edge and course wall (meters)
    min_gap_idx = m_to_idx(min_gap)
    barrier_width = 0.35 + 0.10 * (1-difficulty)   # meters, get wider at low difficulty
    barrier_width_idx = m_to_idx(barrier_width)
    barrier_length = 1.0 + 0.8 * difficulty        # meters, how far barriers extend into course, get longer at high diff
    barrier_length_idx = m_to_idx(barrier_length)
    barrier_height = 0.25 + 0.25 * difficulty      # meters, high enough to force navigating around (not over)
    spacing = (length - 2.5) / (num_barriers + 1)  # distance from each barrier to next (meters)
    spacing_idx = m_to_idx(spacing)

    course_mid_y = m_to_idx(width / 2)
    course_width_idx = m_to_idx(width)

    spawn_x_idx = m_to_idx(1)
    start_buffer_x_idx = m_to_idx(2)
    height_field[:start_buffer_x_idx, :] = 0  # Flat spawn

    # Barrier orientation alternates left/right, with offset so the robot weaves
    left_side = True
    cur_x = start_buffer_x_idx
    goal_spacing = (length - 2) / 7  # Spread goals (even after each barrier)

    # Place barriers
    for i in range(num_barriers):
        barrier_center_x = cur_x + spacing_idx
        barrier_x_start = barrier_center_x
        barrier_x_end = barrier_center_x + barrier_length_idx

        if left_side:
            barrier_y_start = min_gap_idx
            barrier_y_end   = barrier_y_start + barrier_width_idx
            # Goals on far side of barrier, near the right
            goal_y = course_width_idx - min_gap_idx - m_to_idx(0.2)
        else:
            barrier_y_end   = course_width_idx - min_gap_idx
            barrier_y_start = barrier_y_end - barrier_width_idx
            # Goals on far side of barrier, near the left
            goal_y = min_gap_idx + m_to_idx(0.2)

        # Clamp indices just in case
        barrier_y_start = max(0, min(course_width_idx - barrier_width_idx, barrier_y_start))
        barrier_y_end = min(course_width_idx, max(barrier_width_idx, barrier_y_end))

        # Set the barrier height
        height_field[ int(barrier_x_start):int(barrier_x_end), int(barrier_y_start):int(barrier_y_end) ] = barrier_height

        # Place the goal just past the barrier and near the wall, so the agent must go around
        goal_x = barrier_x_end + m_to_idx(0.20)
        goal_x = min(goal_x, m_to_idx(length)-2) # Clamp within terrain
        goals[i+1] = [ goal_x, goal_y ]

        # Step forward for next barrier
        cur_x = barrier_x_end
        left_side = not left_side

    # Start goal (at spawn area, center y)
    goals[0] = [ m_to_idx(1), course_mid_y ]

    # Final goal: Just before the far end of the course, centered y
    goals[7] = [ m_to_idx(length) - m_to_idx(1), course_mid_y ]

    # Ensure first and last meter of course are clear
    height_field[-m_to_idx(1):, :] = 0
    height_field[:start_buffer_x_idx, :] = 0

    return height_field, goals