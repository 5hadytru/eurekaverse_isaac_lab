import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """
    Stepping Stone Balance Course: Series of narrow, staggered raised rectangular stepping stones across a shallow water pit,
    forcing the quadruped to carefully step and balance as it traverses, emphasizing precise foot placement and lateral movement.
    The obstacles are slightly offset laterally and spaced so the robot must weave left/right between goals.
    """

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Terrain parameters
    course_len_idx = m_to_idx(length)
    course_wid_idx = m_to_idx(width)
    spawn_idx_x = m_to_idx(1)
    min_pad = m_to_idx(0.2)

    # Parameters for stepping stones (platforms)
    stone_length = 0.6 + 0.4 * (1 - difficulty)    # [0.6, 1] meters: harder->shorter
    stone_length_idx = m_to_idx(stone_length)
    stone_width = 0.5 + 0.15 * (1 - difficulty)    # [0.5, 0.65] meters: narrow but always sufficient
    stone_width_idx = m_to_idx(stone_width)
    edge_clearance = m_to_idx(0.20)

    pit_depth = -(0.10 + 0.20 * difficulty)   # up to 0.3m at max difficulty, shallow at low difficulty
    stone_height = 0.00 + 0.04 * difficulty   # near ground at easy, up to 4cm above ("floating") at hard

    lateral_offset_range = m_to_idx(0.7) if difficulty > 0.5 else m_to_idx(0.4)
    inter_stone_gap = 0.5 + 0.55 * difficulty   # [0.5, 1.05] meters
    inter_stone_gap_idx = m_to_idx(inter_stone_gap)

    # 1. Set "pit": the central strip of the arena is a water pit, the spawn and ending segments are on solid ground
    height_field[:, :] = 0.     # default
    # Backfill the "water pit"
    pit_start = spawn_idx_x + m_to_idx(0.5)
    pit_end = m_to_idx(length) - m_to_idx(0.5)
    height_field[pit_start:pit_end, :] = pit_depth

    # 2. Set the first and last areas as solid ground
    height_field[0:pit_start, :] = 0
    height_field[pit_end:, :] = 0

    # 3. Place stepping stones in staggered fashion
    # Place the first stone at 1.5 m from start, and then each ~1.0-1.6 m further, alternating their lateral offset
    # Start near the center widthwise, and alternate left-right
    num_stones = 6  # with 1 start, 1 end goal

    stone_xs = [spawn_idx_x + m_to_idx(0.3)]
    gap_variance = m_to_idx(0.15 + 0.3 * difficulty)  # more gap randomness at higher difficulty
    for i in range(1, num_stones):
        prev_x = stone_xs[-1]
        gap = inter_stone_gap_idx + random.randint(-gap_variance, gap_variance)
        new_x = prev_x + stone_length_idx + gap
        if new_x+stone_length_idx >= m_to_idx(length) - edge_clearance:
            break
        stone_xs.append(new_x)

    stone_ys = []
    # Create a gentle stagger: +- lateral_offset within bounds
    center_y = course_wid_idx // 2
    staggering_dir = 1
    for i in range(len(stone_xs)):
        lateral_jitter = (random.randint(-m_to_idx(0.08), m_to_idx(0.08)))  # small randomness for realism
        offset = staggering_dir * (random.randint(int(0.6*lateral_offset_range), lateral_offset_range))
        stone_y = np.clip(center_y + offset + lateral_jitter, edge_clearance, course_wid_idx-edge_clearance)
        stone_ys.append(stone_y)
        staggering_dir *= -1  # alternate left/right

    # 4. Place stones on the height field
    for i, (stone_x, stone_y) in enumerate(zip(stone_xs, stone_ys)):
        x1 = int(max(stone_x - stone_length_idx // 2, pit_start + min_pad))
        x2 = int(min(stone_x + stone_length_idx // 2, pit_end - min_pad))
        y1 = int(max(stone_y - stone_width_idx // 2, edge_clearance))
        y2 = int(min(stone_y + stone_width_idx // 2, course_wid_idx-edge_clearance))
        height_field[x1:x2, y1:y2] = stone_height  # stones rise just above pit
        # Place goal at stone center
        goals[i+1] = [0.5*(x1+x2), 0.5*(y1+y2)]

    # 5. Set precise start and final goals
    # Start goal at spawn on solid ground, centered
    goals[0] = [m_to_idx(0.8), center_y]
    # Final goal is after last stone, on solid ground
    end_x = int(min(stone_xs[-1] + stone_length_idx + 2*inter_stone_gap_idx, course_len_idx-m_to_idx(0.3)))
    goals[-1] = [end_x, center_y + random.randint(-m_to_idx(0.12), m_to_idx(0.12))]

    # 6. Fill extra goals if <8 with linear interpolation along the path
    # This ensures all goals are valid
    idx_of_last = np.count_nonzero(np.any(goals != 0, axis=1)) - 1
    if idx_of_last < 7:
        for j in range(idx_of_last+1, 8):
            v = (j-idx_of_last)/(8-idx_of_last)
            goals[j] = (1-v)*goals[idx_of_last] + v*goals[-1]

    return height_field, goals