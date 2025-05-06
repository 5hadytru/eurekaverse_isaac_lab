import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of urban-style curb jumps: robot must cross raised curbs ("speed bumps") of varying heights and step lengths, testing dynamic balance and stepping precision."""
    
    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]
    
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for obstacles ("curbs" or "bumps")
    curb_height_min = 0.06 + 0.10 * difficulty         # 6-16 cm tall
    curb_height_max = 0.12 + 0.16 * difficulty         # 12-28 cm tall
    curb_width_min = 1.1                               # at least quadruped width + margin
    curb_width_max = 2.1
    curb_length = 0.18 + 0.22 * difficulty             # from 18 to 40 cm "thick" speed-bump
    curb_spacing_min = 0.8 + 1.1 * (1-difficulty)      # at harder diff, less space between obstacles
    curb_spacing_max = 1.5 + 2.0 * (1-difficulty)
    curb_overall_steps = 7                             # Number of obstacles

    mid_y = m_to_idx(width) // 2

    def add_curb(x0, y_center, curb_len, curb_width, curb_height):
        half_width = curb_width // 2
        x1 = int(x0)
        x2 = int(x0 + curb_len)
        y1 = int(y_center - half_width)
        y2 = int(y_center + half_width)
        y1 = max(0, y1)
        y2 = min(m_to_idx(width), y2)
        x2 = min(m_to_idx(length), x2)
        height_field[x1:x2, y1:y2] = curb_height

    # Start: spawn on "ground"
    spawn_length = m_to_idx(2)
    height_field[0:spawn_length, :] = 0        # Flat ground for safe spawning
    # First goal: just in front of first curb
    goals[0] = [m_to_idx(1.0), mid_y]

    # Position cursor for first curb after spawn zone
    x_cur = spawn_length + m_to_idx(0.2)
    curb_locs = []
    
    for i in range(curb_overall_steps):
        # Random widths, lengths, heights within bounds for variety
        curb_width = random.uniform(curb_width_min, curb_width_max)
        curb_len = random.uniform(curb_length * 0.7, curb_length * 1.25)
        curb_height = random.uniform(curb_height_min, curb_height_max)
        # Place curb
        add_curb(x_cur, mid_y, m_to_idx(curb_len), m_to_idx(curb_width), curb_height)
        curb_locs.append((x_cur + m_to_idx(curb_len)//2, mid_y))
        # Place goal on curb top midline, offset slightly to encourage straight direction 
        goals[i+1] = [x_cur + m_to_idx(curb_len)//2, mid_y]
        # Advance: set next curb spacing
        space = random.uniform(curb_spacing_min, curb_spacing_max)
        x_cur += m_to_idx(curb_len + space)

        # Avoid running over course boundary
        if x_cur >= m_to_idx(length) - m_to_idx(2):
            break  # Don't overrun the course

    # Last goal: at course end, centerline, after all curbs
    goals[7] = [min(m_to_idx(length)-m_to_idx(1.0), x_cur), mid_y]

    # Final flat ground patch at end for recovery (ensure no obstacle at very end)
    end_start = int(min(x_cur, m_to_idx(length) - m_to_idx(1.0)))
    height_field[end_start:, :] = 0
    
    # Ensure all goals in-bounds (replace missing goals with in-place copies of last valid goal)
    for j in range(len(curb_locs)+1, 8):
        goals[j] = goals[j-1]

    return height_field, goals