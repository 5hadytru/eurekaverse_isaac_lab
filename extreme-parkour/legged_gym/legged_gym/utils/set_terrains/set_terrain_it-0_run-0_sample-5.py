import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Slalom-style alternating tall barriers for robot to weave through, testing agile turning and path planning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Obstacle settings
    # High walls spaced such that the robot must weave left and right to avoid collision
    wall_length = 0.3 + 0.3 * difficulty       # Walls get longer at higher difficulty (block more of course)
    wall_thickness = 0.16                      # Always at least 16 cm thick (about 0.6x body width)
    wall_height = 0.25 + 0.25 * difficulty     # Up to 0.5 m high (need to go around not over for this task)
    gap_width = 0.6 + (1.1 - 0.8 * difficulty) # Space between wall and course boundary (robot must fit through)
    midline_y = m_to_idx(width / 2)
    
    n_walls = 6
    # Minimum spacing between walls to prevent overlap and ensure 8 goals fit
    min_spawn_x = m_to_idx(2)
    course_end_x = m_to_idx(length - 0.8)
    wall_spacing = (course_end_x - min_spawn_x) // n_walls
    # Adjust for field quantization
    wall_length_idx = m_to_idx(wall_length)
    wall_thickness_idx = m_to_idx(wall_thickness)
    gap_width_idx = m_to_idx(gap_width)
    
    # Set flat spawn region (no obstacles in first 2 meters)
    height_field[0:min_spawn_x, :] = 0.0

    # Place alternating barriers and goals
    left = True
    goal_idx = 0
    for i in range(n_walls):
        wall_x = min_spawn_x + i * wall_spacing
        # Walls alternate left/right leaving a "gap" before wall
        if left:
            wall_y1 = 0
            wall_y2 = midline_y - gap_width_idx // 2
            # Place wall up to the (gap)
            height_field[wall_x:wall_x + wall_length_idx, wall_y1:wall_y2] = wall_height
            # Place goal in the gap, slightly before the wall
            goal_x = wall_x - m_to_idx(0.25)
            goal_y = wall_y2 + gap_width_idx // 3
        else:
            wall_y1 = midline_y + gap_width_idx // 2
            wall_y2 = m_to_idx(width)
            height_field[wall_x:wall_x + wall_length_idx, wall_y1:wall_y2] = wall_height
            goal_x = wall_x - m_to_idx(0.25)
            goal_y = wall_y1 - gap_width_idx // 3
        # Clamp to valid indices
        goal_x = np.clip(goal_x, min_spawn_x, m_to_idx(length) - 1)
        goal_y = np.clip(goal_y, 0, m_to_idx(width) - 1)
        goals[goal_idx] = [goal_x, goal_y]
        goal_idx += 1
        left = not left

    # Place 7th goal at the center near the end, after the last wall, to force a straight path to the end
    last_gap_x = min_spawn_x + n_walls * wall_spacing - m_to_idx(0.2)
    last_gap_x = np.clip(last_gap_x, 0, m_to_idx(length) - 1)
    last_gap_y = midline_y
    goals[6] = [last_gap_x, last_gap_y]

    # Final goal is at the very end, to drive robot to completion
    goals[7] = [m_to_idx(length) - m_to_idx(0.5), midline_y]
    # Make sure there are no obstacles at last 50cm, robot can finish on flat ground
    height_field[m_to_idx(length)-m_to_idx(0.6):, :] = 0.0

    # Ensure all 8 goals are within map bounds (safety check)
    goals = np.clip(goals, [[0,0]], [m_to_idx(length)-1, m_to_idx(width)-1])
    
    return height_field, goals