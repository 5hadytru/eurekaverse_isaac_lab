import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone course: The robot must walk across a series of narrow, staggered, flat-topped blocks ('stepping stones') over a sunken pit, testing precise limb placement and gait adaptation for lateral and forward motion."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))  # [x, y] format in quantized indices

    course_length = m_to_idx(length)
    course_width = m_to_idx(width)
    spawn_length = m_to_idx(2.0)
    mid_y = course_width // 2

    # Course parameters: stepping stone settings
    stone_width = max(0.45, 0.7 - 0.4 * difficulty)         # meters, at least 0.45m wide (wider at easy, narrow at hard)
    stone_length = max(0.5, 0.8 - 0.3 * difficulty)         # meters, at least 0.5m long
    stone_height = 0.02 + 0.13 * difficulty                 # meters, low at easy, up to 15cm at hard
    stone_gap = 0.55 + 0.55 * difficulty                    # meters, forward gap between stones
    lateral_offset_max = 0.17 + 0.23 * difficulty           # meters, how far side-to-side the stones are staggered

    # Quantize values
    stone_width_idx = m_to_idx(stone_width)
    stone_length_idx = m_to_idx(stone_length)
    stone_gap_idx = m_to_idx(stone_gap)
    lateral_offset_idx = m_to_idx(lateral_offset_max)

    # Create a pit (negative height) across most of the course
    height_field[spawn_length:,:] = -0.6 - 0.2 * difficulty   # lower at higher difficulty (force stepping)
    # Make the spawn area flat for a safe start
    height_field[:spawn_length,:] = 0.

    # Place initial goal (on flat ground)
    goals[0] = [m_to_idx(1.0), mid_y]

    # Plan stones: start at the end of the spawn zone, up to near the end of the course
    n_stones = 4           # 4 stepping stones = 5 goals incl. start and finish
    first_stone_x = spawn_length + m_to_idx(0.7)    # start the first stone just after spawn
    stone_xs = []
    stone_ys = []

    # Alternate stone y positions to make the robot zigzag: left/right of midline
    for i in range(n_stones):
        x_pos = first_stone_x + i * (stone_gap_idx + stone_length_idx)
        if x_pos + stone_length_idx > course_length - m_to_idx(0.8):
            x_pos = course_length - m_to_idx(1.5)
        # Lateral staggering: alternate left and right of midline, increase offset on harder difficulty
        if i % 2 == 0:
            y_offset = -lateral_offset_idx
        else:
            y_offset = lateral_offset_idx
        # Random small jitter within reasonable bounds, to avoid excessive regularity
        y_pos = mid_y + y_offset + random.randint(-m_to_idx(0.1), m_to_idx(0.1))
        # Clamp position to fit within course width and ensure stone is at least on the field
        y_pos = np.clip(y_pos, stone_width_idx//2, course_width - stone_width_idx//2)
        stone_xs.append(x_pos)
        stone_ys.append(y_pos)

        # Draw the stone block
        x1 = int(x_pos)
        x2 = int(x_pos + stone_length_idx)
        y1 = int(y_pos - stone_width_idx // 2)
        y2 = int(y_pos + stone_width_idx // 2)
        height_field[x1:x2, y1:y2] = stone_height

        # Place a goal at the center of the stone
        goals[i+1] = [(x1 + x2)//2, (y1 + y2)//2]

    # Final patch of walkable flat ground after last stone as runout for robot stop
    final_floor_x = x2 + m_to_idx(0.7)
    if final_floor_x < course_length:
        height_field[final_floor_x:, :] = 0.

        # Final goal at end of course (middle in y)
        goals[4] = [min(final_floor_x + m_to_idx(0.7), course_length-1), mid_y]
    else:
        goals[4] = [course_length-1, mid_y]

    return height_field, goals