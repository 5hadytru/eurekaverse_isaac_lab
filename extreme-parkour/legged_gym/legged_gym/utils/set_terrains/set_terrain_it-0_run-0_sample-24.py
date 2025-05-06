import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Eight stepping stones over a narrow trench to test the robot's precise foot placement and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))  # 8 goals

    # Stepping stone/trench design parameters
    spawn_length = m_to_idx(2)  # Flat ground before first obstacle
    height_field[:spawn_length, :] = 0  # Keep spawn area flat

    # Define trench dimensions (width = 1.4m, length = spans most of course after spawn)
    trench_x_start = spawn_length
    trench_x_end = m_to_idx(length - 2)              # leave a finish area at the end
    trench_width = m_to_idx(1.4)
    trench_center = m_to_idx(width // 2)
    trench_y1 = trench_center - trench_width//2
    trench_y2 = trench_center + trench_width//2

    # Trench depth
    trench_depth = -0.65 - 0.35*difficulty

    height_field[trench_x_start:trench_x_end, trench_y1:trench_y2] = trench_depth

    # Stepping stone parameters
    stone_length = m_to_idx(0.6)
    stone_width = m_to_idx(0.35)
    stone_height = 0.0   # flush with spawn area (forces careful foot placement — no step-up)
    # Spacing: stones separated to force single step, and zig-zag with some lateral offset

    x_positions = np.linspace(trench_x_start + m_to_idx(0.7), trench_x_end - m_to_idx(1.2), 8).astype(np.int16)
    y_mid = m_to_idx(width / 2)
    max_offset = m_to_idx(0.4 + 0.3 * difficulty)     # Range of lateral offsets for zig-zag

    # Alternating left/right offsets for zig-zag
    offsets = []
    for i in range(8):
        # Odd stones offset left, even stones offset right
        sign = (-1) ** i
        # At higher difficulty, increase offset magnitude and some random
        base_offset = int(sign * (max_offset * (0.45 + 0.4*random.random())))
        offsets.append(base_offset)

    for i, (x, offset) in enumerate(zip(x_positions, offsets)):
        y = y_mid + offset
        # Keep stone within trench edges
        y = np.clip(y, trench_y1 + stone_width//2, trench_y2 - stone_width//2)
        x1 = int(x - stone_length//2)
        x2 = int(x + stone_length//2)
        y1 = int(y - stone_width//2)
        y2 = int(y + stone_width//2)
        # Stepping stone slightly higher at higher difficulties (to force more precision)
        stone_elev = stone_height + difficulty*0.07 + 0.03*random.uniform(-1,1)
        height_field[x1:x2, y1:y2] = stone_elev
        goals[i] = [int((x1 + x2)/2), int((y1 + y2)/2)]

    # Start (spawn) and finish areas
    # Place goal[0] at start area, before the first stone
    goals[0] = [m_to_idx(1.5), y_mid]
    # Remaining 7 goals: place on each stepping stone (the above loop has set those)
    # Last goal: after last stone, in flat finish area
    finish_x = m_to_idx(length - 1.0)
    finish_y = y_mid + offsets[-1]   # Continue the zig-zag
    finish_y = np.clip(finish_y, m_to_idx(0.5), m_to_idx(width - 0.5))
    goals[-1] = [finish_x, finish_y]
    height_field[trench_x_end:, :] = 0  # finish area is flat

    # Ensure all goals are within bounds
    goals = np.clip(goals, [0,0], [m_to_idx(length)-1, m_to_idx(width)-1])

    return height_field, goals