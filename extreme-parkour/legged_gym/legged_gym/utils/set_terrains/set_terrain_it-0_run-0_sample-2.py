import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone sequence: Tests the robot's precise stepping & agile turning via narrow, spaced stepping stones across a pit."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    mid_y = m_to_idx(width // 2)
    spawn_length = m_to_idx(2)
    total_length = m_to_idx(length)
    total_width = m_to_idx(width)
    
    # 1. Set starting area flat
    height_field[0:spawn_length, :] = 0
    goals[0] = [m_to_idx(1.0), mid_y]  # slightly ahead of center spawn

    # 2. Create a pit across the width with raised, spaced stepping stones
    pit_start_x = spawn_length
    pit_end_x = m_to_idx(length - 1)
    pit_depth = -0.70 - 0.2 * difficulty  # force to use stepping stones

    height_field[pit_start_x:pit_end_x, :] = pit_depth

    # 3. Stepping stone parameters --- width just over body width, length = 0.5m, gaps adjust with difficulty
    stone_length = 0.5
    stone_width = 0.35 + 0.15 * (1 - difficulty)  # wider at low difficulty, minimum always larger than one footstep
    n_stones = 6
    step_gap = 0.40 + difficulty * 0.32  # from 40cm up to 72cm gaps
    stone_height = 0.02 + 0.06 * difficulty  # higher at high difficulty

    first_stone_x = spawn_length + int(m_to_idx(0.5))
    y_stride = m_to_idx(0.90 - 0.5 * difficulty)  # later stones may require turning

    # Pre-calculate stone location centers for stones 1-6
    stone_positions = []
    cur_x = first_stone_x
    cur_y = mid_y
    for i in range(n_stones):
        # Stagger left-right so the quadruped must steer
        if i % 2 == 1:
            delta_y = y_stride
        else:
            delta_y = -y_stride
        center_y = cur_y + delta_y if i > 0 else cur_y
        stone_positions.append((cur_x, center_y))
        cur_x += m_to_idx(step_gap)
        cur_y = center_y

    # 4. Place the stones and set goal points over each stone
    for i, (sx, sy) in enumerate(stone_positions):
        # Top left and bottom right bounds in indices
        half_len = m_to_idx(stone_length/2)
        half_wid = m_to_idx(stone_width/2)
        x1 = max(0, sx - half_len)
        x2 = min(total_length, sx + half_len)
        y1 = max(0, sy - half_wid)
        y2 = min(total_width, sy + half_wid)
        height_field[x1:x2, y1:y2] = stone_height
        # Goals
        goals[i+1] = [sx, sy]

    # 5. After last stone, transition to a 'safe zone' at raised ground, with final goal
    final_zone_x = min(total_length, stone_positions[-1][0] + m_to_idx(step_gap//2))
    height_field[final_zone_x:, :] = 0
    goals[7] = [final_zone_x + m_to_idx(0.8), mid_y]  # put final goal in the middle of the far flat

    return height_field, goals