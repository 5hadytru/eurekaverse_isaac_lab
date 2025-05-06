import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone obstacle course: sequence of evenly-spaced narrow flat stepping blocks crossing a deep trench."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Key parameters (all lengths in meters)
    course_length = length
    course_width = width

    spawn_x = 1.0                   # Meters, spawn area, flat ground
    spawn_length = m_to_idx(spawn_x)
    trench_depth = -(0.4 + 0.8 * difficulty)  # meters below zero
    trench_start = m_to_idx(2.0)    # Meters, trench after spawn
    trench_end = m_to_idx(length - 0.8)
    trench_mid_y = m_to_idx(course_width // 2)
    trench_width = m_to_idx(course_width)

    # Stepping stone parameters
    stone_length = max(0.4, 0.6 - 0.1 * difficulty)                    # 0.4 ~ 0.6 m
    stone_width = max(0.4, 0.5 - 0.1 * difficulty)                     # 0.4 ~ 0.5 m
    stone_height = 0.0  # Stones are level with spawn area
    n_stones = 6

    min_gap = 0.3 + 0.5 * difficulty     # gap between stones (meters)
    max_gap = 0.4 + 0.8 * difficulty     # in meters

    mid_y = m_to_idx(width / 2)
    lane_offset = 0  # stones placed in straight line

    # 1. Set spawn area (flat ground)
    height_field[:trench_start,:] = 0

    # 2. Set trench (deep pit, robot must cross via stones)
    height_field[trench_start:trench_end, :] = trench_depth

    # Helper: place a stone and associated goal
    def place_stepping_stone(stone_idx, cur_x):
        # Stones are perpendicular to x, aligned with the center lane
        x1 = m_to_idx(cur_x - stone_length/2)
        x2 = m_to_idx(cur_x + stone_length/2)
        y1 = mid_y - (m_to_idx(stone_width)//2)
        y2 = mid_y + (m_to_idx(stone_width)//2)
        # Place stone at ground height (0), edges clipped if out of bounds
        x1 = max(trench_start, x1)
        x2 = min(m_to_idx(length), x2)
        y1 = max(0, y1)
        y2 = min(m_to_idx(width), y2)
        height_field[x1:x2, y1:y2] = stone_height
        # Place goal roughly at stone center
        goals[stone_idx] = [int(0.5*(x1+x2)), int(0.5*(y1+y2))]

    # 3. Place stepping stones and goals
    # Evenly distribute stones across the trench
    trench_distance = trench_end - trench_start
    stone_centers = np.linspace(
        trench_start + m_to_idx(stone_length/2 + 0.05),
        trench_end - m_to_idx(stone_length/2 + 0.05),
        n_stones,
    )

    # Place first goal at the end of the spawn zone
    goals[0] = [trench_start-m_to_idx(0.3), mid_y]  # safe edge before trench

    for i, stone_center in enumerate(stone_centers):
        place_stepping_stone(i+1, stone_center * field_resolution)

    # 4. Final recovery platform and goal on the far side of the trench
    recovery_start = trench_end
    recovery_end = m_to_idx(length)
    height_field[recovery_start:recovery_end, :] = 0
    goals[7] = [int((recovery_start + recovery_end)//2), mid_y]

    return height_field, goals