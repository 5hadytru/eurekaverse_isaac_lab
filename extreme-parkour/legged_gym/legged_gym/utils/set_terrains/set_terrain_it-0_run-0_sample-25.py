import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone trail: raised, offset stone slabs that require lateral and longitudinal foot placement accuracy."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))
    
    # Step stone parameters (size depends on quadruped size and difficulty)
    robot_length, robot_width = 0.645, 0.28
    stone_length = 0.7 + 0.2 * (1 - difficulty)       # At least 0.7m long
    stone_width  = 0.55 + 0.3 * (1 - difficulty)      # At least wider than robot width
    stone_height = 0.15 + 0.15 * difficulty           # Stones grow taller with difficulty
    gap = 0.30 + 0.45 * difficulty                    # Gap between stones grows with difficulty
    
    stone_length_idx = m_to_idx(stone_length)
    stone_width_idx  = m_to_idx(stone_width)
    gap_idx          = m_to_idx(gap)

    spawn_x = m_to_idx(1)  # spawn at x = 1m
    spawn_y = m_to_idx(width / 2)
    height_field[:spawn_x+1,:] = 0  # spawn area

    # Place initial goal in spawn area
    goals[0] = [spawn_x, spawn_y]
    
    # The path: alternate step stones left and right about a center track
    step_n = 7
    offset_mag = int(m_to_idx(0.65 + 0.1 * difficulty))  # How far each stone may be offset from center
    center_y = spawn_y
    cur_x = spawn_x + m_to_idx(0.6)
    
    for i in range(step_n):
        # Alternate left-right offset, but ensure within the field
        dir_ = -1 if i % 2 == 0 else 1
        # Move further off-center as difficulty increases
        offset_y = dir_ * (offset_mag - random.randint(0, m_to_idx(0.2)))
        y_c = int(np.clip(center_y + offset_y, stone_width_idx//2, m_to_idx(width) - stone_width_idx//2 - 1))
        
        # Stone bounds
        x1 = int(cur_x)
        x2 = int(np.clip(cur_x + stone_length_idx, 0, m_to_idx(length)))
        y1 = int(y_c - stone_width_idx // 2)
        y2 = int(y_c + stone_width_idx // 2)
        # Ensure within bounds
        x1, x2 = max(x1, 0), min(x2, m_to_idx(length))
        y1, y2 = max(y1, 0), min(y2, m_to_idx(width))
        
        # Place the stone: set raised height
        hf_height = stone_height * (0.9 + 0.2 * random.random())  # some variation
        height_field[x1:x2, y1:y2] = hf_height

        # Goal on this stone, slightly random within stone area
        goals[i+1] = [ (x1 + x2)//2, int(np.clip((y1 + y2)//2 + random.randint(-2,2), 0, m_to_idx(width)-1))]
        
        # Move to next stone position
        cur_x = x2 + gap_idx

    # Surround the stepping stones with pits: set all other unraised areas to -0.7m (except spawn area)
    pit_height = -0.7 - 0.3 * difficulty  # Deeper pit at higher difficulty
    field_len = m_to_idx(length)
    for x in range(spawn_x+1, field_len):
        for y in range(m_to_idx(width)):
            # If not part of a step stone
            if height_field[x,y] < 0.01:
                height_field[x,y] = pit_height

    # Final goal in flat safe zone at end of course
    safezone_len = m_to_idx(1.0)
    end_x = field_len - safezone_len
    height_field[end_x:, :] = 0
    goals[-1] = [end_x + safezone_len//2, center_y]

    return height_field, goals