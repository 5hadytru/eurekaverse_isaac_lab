import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stones: Repeated narrow, slightly offset, raised blocks across a pond for precise foot placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # Constants for obstacle construction
    spawn_x = m_to_idx(1)
    pond_start_x = m_to_idx(2)             # Area after which obstacles begin
    pond_end_x = m_to_idx(length-1.0)      # Keep margins at both ends
    course_width_idx = m_to_idx(width)
    stone_min_length, stone_max_length = 0.45, 0.7
    stone_min_width, stone_max_width = 0.4, 0.6
    stone_min_h, stone_max_h = 0.04, 0.12

    # Scale above to difficulty
    n_stones = 5 + int(difficulty * 4)     # 5 to 9 stones
    pond_depth = 0.10 + 0.4*difficulty     # "Water"/pit is deeper at high difficulty

    # Build "water" (pit) from pond_start_x to pond_end_x
    height_field[pond_start_x:pond_end_x, :] = -pond_depth

    # Generate stepping stone centers and dims
    step_xs = np.linspace(pond_start_x + m_to_idx(0.7), pond_end_x - m_to_idx(0.8), n_stones)
    step_xs = [int(round(x)) for x in step_xs]

    mid_y = course_width_idx // 2
    max_offset = m_to_idx(0.7 + (0.4 * difficulty))  # Stones can be offset by up to 0.7-1.1 meters at high difficulty
    stone_ys = [mid_y]
    for i in range(1, n_stones):
        prev = stone_ys[-1]
        offset = int(round(random.uniform(-max_offset, max_offset)))
        # constrain y so that stone stays within bounds
        new_y = min(max(course_width_idx//4, prev + offset), 3*course_width_idx//4)
        stone_ys.append(new_y)
    # Insert the first stepping stone just in front of the pond, for a smooth entry
    stone_ys[0] = mid_y

    # Place stepping stones
    stone_rects = []
    for i, (stone_x, stone_y) in enumerate(zip(step_xs, stone_ys)):
        # Size tightens as difficulty increases
        l = stone_min_length + (stone_max_length-stone_min_length)*(1-difficulty)
        w = stone_min_width + (stone_max_width-stone_min_width)*(1-difficulty)
        # Slightly randomize size
        l += random.uniform(-0.08, 0.08)
        w += random.uniform(-0.08, 0.08)
        l_idx = max(m_to_idx(l), m_to_idx(0.4))
        w_idx = max(m_to_idx(w), m_to_idx(0.4))
        h = stone_min_h + (stone_max_h-stone_min_h)*difficulty + random.uniform(0.00, 0.08)
        # Place stone rectangle (clamp within boundaries)
        x1 = max(stone_x-l_idx//2, pond_start_x)
        x2 = min(stone_x+l_idx//2, m_to_idx(length))
        y1 = max(stone_y-w_idx//2, 0)
        y2 = min(stone_y+w_idx//2, course_width_idx)
        height_field[x1:x2, y1:y2] = h
        stone_rects.append(((x1, x2), (y1, y2)))
    
    # Place first and last "platforms" (for start/finish outside pond)
    dock_w = m_to_idx(2.0)
    dock_l = m_to_idx(0.8)
    # Start platform:
    x1 = spawn_x - dock_l if (spawn_x - dock_l) > 0 else 0
    x2 = pond_start_x
    y1 = (mid_y - dock_w//2)
    y2 = (mid_y + dock_w//2)
    height_field[x1:x2, y1:y2] = 0
    # End platform:
    x1 = pond_end_x
    x2 = m_to_idx(length)
    height_field[x1:x2, y1:y2] = 0

    # Set goals:
    # [0] Spawn point, [1-3] at stones 1, n//2, n-1, [4] at course end
    goals[0] = [float(spawn_x+m_to_idx(0.5)), float(mid_y)]
    goals[1] = [float(step_xs[1]), float(stone_ys[1])]
    goals[2] = [float(step_xs[n_stones//2]), float(stone_ys[n_stones//2])]
    goals[3] = [float(step_xs[-2]), float(stone_ys[-2])]
    # Final goal on end platform, at center
    goals[4] = [float((pond_end_x + x2)//2), float(mid_y)]

    return height_field, goals