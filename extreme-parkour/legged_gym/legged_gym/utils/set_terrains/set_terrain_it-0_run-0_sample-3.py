import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Five staggered low walls the robot must step over at angles, testing precise foot placement and limb high-stepping."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]
    
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # Wall parameters
    n_walls = 5
    wall_length = 1.4 - 0.25 * difficulty   # walls get shorter as difficulty increases (more precise steps)
    wall_width = 0.25 + 0.25 * difficulty   # walls get wider as difficulty increases (can't step around)
    wall_height = 0.08 + 0.15 * difficulty  # up to 0.23 m step
    
    gap_between = 1.05 + 0.3 * (1-difficulty)   # closer together at higher diff
    wall_clearance_from_edge = 0.7

    wall_length_idx = m_to_idx(wall_length)
    wall_width_idx = m_to_idx(wall_width)
    gap_between_idx = m_to_idx(gap_between)
    wall_clearance_idx = m_to_idx(wall_clearance_from_edge)
    course_length_idx = m_to_idx(length)
    course_width_idx = m_to_idx(width)

    # Offsets to stagger the walls side to side (positive and negative y-offsets alternately)
    max_offset = m_to_idx(0.9)  # keeps wall inside course
    stagger_seq = [0, max_offset, -max_offset, max_offset, -max_offset]  # zigzag
    
    # The x position of the first wall, make sure not to interfere with spawn area at x=0:2.0m
    x0 = m_to_idx(2.2)

    def place_wall(center_x, center_y, wall_idx):
        """Places a wall with its center at (center_x, center_y)"""
        x1 = int(center_x - wall_length_idx//2)
        x2 = int(center_x + wall_length_idx//2)
        y1 = int(max(center_y - wall_width_idx//2, 0))
        y2 = int(min(center_y + wall_width_idx//2, course_width_idx-1))
        height_field[x1:x2, y1:y2] = wall_height
        
        # Mark the goal after the wall (a little past the wall in x, at the same y)
        gx = min(x2 + m_to_idx(0.37), course_length_idx-1)
        goals[wall_idx] = [gx, int(center_y)]

    # Flat start platform
    height_field[:x0, :] = 0.0
    goals[0] = [m_to_idx(1.0), course_width_idx // 2]  # initial goal halfway down spawn zone

    cur_x = x0
    mid_y = course_width_idx // 2

    for i in range(1, n_walls+1):
        # Stagger (zig-zag) the wall left and right from center
        stagger_y = mid_y + stagger_seq[i-1]
        place_wall(cur_x, stagger_y, i if i<5 else 4)  # clamp idx for the last one
        
        # The gap leaves enough space for the robot to step down after the wall
        cur_x += wall_length_idx + gap_between_idx

    # Ensure remaining ground after last wall is flat for terminal reward
    x_end = min(cur_x + m_to_idx(1.2), course_length_idx)
    height_field[cur_x:x_end, :] = 0.0
    # Final goal is well past last wall, on centerline
    goals[4] = [min(x_end-1, course_length_idx-1), mid_y]

    return height_field, goals