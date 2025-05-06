import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Alternating balance beams and 'stepping stone' pads to test precise foot placement and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course parameters
    mid_y = m_to_idx(width / 2)
    spawn_length = m_to_idx(2)
    length_idx = m_to_idx(length)
    width_idx = m_to_idx(width)

    # Balance beam setup
    beam_length = 1.6 + 0.8 * difficulty     # Longer beams at higher difficulty
    beam_width = 0.26 + 0.08 * (1-difficulty) # Beams get narrower as difficulty increases (down to 0.26m)
    beam_height = 0.08 + 0.10 * difficulty   # Higher beams at harder difficulty

    # Stepping stones setup
    stone_count = int(3 + 2*difficulty)         # More stones at higher difficulty
    stone_diameter = 0.37 - 0.08 * difficulty   # Smaller stones at higher difficulty, never below 0.29m
    stone_height = 0.10 + 0.08 * difficulty

    dx_between = 1.1 + 0.12 * difficulty   # Distance between obstacles

    # Helper to add a beam perpendicularly or straight
    def add_beam(start_x_idx, y_idx, orientation='horizontal'):
        l_idx = m_to_idx(beam_length)
        w_idx = m_to_idx(beam_width)
        h = beam_height
        if orientation == 'horizontal':
            y1 = int(y_idx - w_idx//2)
            y2 = int(y_idx + w_idx//2)
            x2 = min(start_x_idx + l_idx, length_idx)
            height_field[start_x_idx:x2, y1:y2] = h
            # Center goal
            goals_list.append([start_x_idx + l_idx // 2, y_idx])
            return x2
        elif orientation == 'vertical':
            x1 = int(start_x_idx - w_idx//2)
            x2 = int(start_x_idx + w_idx//2)
            y2 = min(y_idx + l_idx, width_idx)
            height_field[x1:x2, y_idx:y2] = h
            # Center goal
            goals_list.append([start_x_idx, y_idx + l_idx // 2])
            return y2

    # Helper to add stepping stones in a line
    def add_stepping_stones(start_x_idx, y_idx, count, step_dx):
        s_idx = m_to_idx(stone_diameter)
        half = s_idx//2
        for i in range(count):
            cx = start_x_idx + i*step_dx
            y1 = int(y_idx - half)
            y2 = int(y_idx + half)
            x1 = int(cx - half)
            x2 = int(cx + half)
            # Only place inside bounds:
            if 0 <= x1 < length_idx and 0 < y1 < width_idx and x2 < length_idx and y2 < width_idx:
                # Circular mask for stepping stones
                for x in range(x1, x2):
                    for y in range(y1, y2):
                        if (x-cx)**2 + (y-y_idx)**2 < (half**2):
                            height_field[x,y] = stone_height 
                # Place a goal on the center of the middle stone
                if i == (count // 2):
                    goals_list.append([cx, y_idx])
        return start_x_idx + count*step_dx

    # 1. Flat spawn zone
    height_field[:spawn_length, :] = 0
    goals_list = []
    # First goal at spawn
    goals_list.append([spawn_length - m_to_idx(0.5), mid_y])

    # 2. First beam (horizontal, center)
    cur_x = m_to_idx(2.5)
    cur_y = mid_y
    cur_x = add_beam(cur_x, cur_y, orientation='horizontal')
    cur_x += m_to_idx(dx_between)

    # 3. First stepping stones (center, straight)
    stone_dx = m_to_idx(0.45 + 0.10 * difficulty)
    cur_x = add_stepping_stones(cur_x, cur_y, stone_count, stone_dx)
    cur_x += m_to_idx(dx_between)

    # 4. Second beam (horizontal, center but shifted y, for a turn)
    y_shift = m_to_idx(0.90 + 0.80 * difficulty) * random.choice([-1, 1])
    cur_y = np.clip(mid_y + y_shift, m_to_idx(0.8), width_idx - m_to_idx(0.8))
    cur_x = add_beam(cur_x, cur_y, orientation='horizontal')
    cur_x += m_to_idx(dx_between * 0.7)

    # 5. Second stepping stones, angled path toward original midline
    step_vec = np.array([np.cos(np.pi/7), np.sin(np.pi/7)])  # Gentle diag
    cx = cur_x
    cy = cur_y
    for i in range(stone_count):
        ix = int(cx)
        iy = int(cy)
        s_idx = m_to_idx(stone_diameter)
        half = s_idx//2
        for x in range(ix-half, ix+half):
            for y in range(iy-half, iy+half):
                if (x-ix)**2 + (y-iy)**2 < (half**2):
                    if 0 <= x < length_idx and 0 <= y < width_idx:
                        height_field[x, y] = stone_height
        if i == (stone_count // 2):
            goals_list.append([ix, iy])
        cx += stone_dx * step_vec[0]
        cy -= (y_shift / (stone_count-1))
    cur_x = int(cx + m_to_idx(0.15))

    # 6. Third beam, back to midline
    cur_y = mid_y
    cur_x = add_beam(cur_x, cur_y, orientation='horizontal')
    cur_x += m_to_idx(dx_between * 0.7)

    # 7. Final stepping stones to finish line (midline, straight)
    cur_x = add_stepping_stones(cur_x, cur_y, stone_count, stone_dx)
    cur_x += m_to_idx(0.5)

    # 8. Final goal at course end
    final_goal_x = min(cur_x, length_idx - m_to_idx(1.2))
    goals_list.append([final_goal_x, mid_y])

    # If fewer than 8 goals placed, repeat last goal
    for i, g in enumerate(goals_list[:8]):
        goals[i] = g
    if len(goals_list) < 8:
        goals[len(goals_list):] = goals_list[-1]

    # Make everything outside beams/stones a "pit"
    # (Everywhere height==0 and x >= spawn zone, except beams/stones, set to -0.7)
    pit_mask = (height_field == 0)
    pit_mask[:spawn_length, :] = False  # Leave spawn flat
    height_field[pit_mask] = -0.7 - 0.2 * difficulty  # Deeper pit at harder difficulty

    return height_field, goals