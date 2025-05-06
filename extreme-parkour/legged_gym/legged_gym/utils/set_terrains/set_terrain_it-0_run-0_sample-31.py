import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Zig-zag set of wide balance beams requiring precise foot placement for stable walking."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for balance beams
    beam_width = 1.0 - 0.4 * difficulty   # 1.0m at easy, 0.6m at hard (must be at least 0.4m)
    beam_width = max(beam_width, 0.6)
    beam_length = 2.0                    # Length of each segment in meters
    beam_height = 0.1 + 0.18 * difficulty # Height above ground
    gap_length = 0.16 + 0.34 * difficulty # Gaps between beams
    gap_height = -1.5                     # Pit below beams so robot falls off if unstable

    mid_y = m_to_idx(width) // 2

    # Alternate beams: left, right, left... with goals at ends, forming a zig-zag path
    # Left and right offsets
    beam_lateral_offset = 0.75 + 0.50 * difficulty  # maximum lateral distance from center
    n_beams = 7  # 7 beams, 1 flat finish area

    start_x = m_to_idx(2.0) # spawn behind (must leave 2m for spawn area)
    cur_x = start_x

    # Set spawn area to flat ground
    height_field[:start_x, :] = 0
    goals[0] = [start_x - m_to_idx(0.5), mid_y]  # initial goal: right ahead

    # Set rest of terrain to be a pit by default
    height_field[start_x:, :] = gap_height

    for i in range(n_beams):
        # Decide left or right (alternate beams)
        side = -1 if i % 2 == 0 else 1  # start left, then right, then left...
        offset_m = side * (beam_lateral_offset / 2 + random.uniform(-0.1, 0.1) * (1-difficulty))
        beam_mid_y = mid_y + m_to_idx(offset_m)

        # Ensure beam stays in-bounds
        min_y = m_to_idx(max(0.0, (width/2) + offset_m - beam_width/2))
        max_y = m_to_idx(min(width, (width/2) + offset_m + beam_width/2))
        min_y = max(0, min_y)
        max_y = min(m_to_idx(width), max_y)

        # Beam X range
        beam_start_x = cur_x
        beam_end_x = cur_x + m_to_idx(beam_length)
        beam_end_x = min(beam_end_x, m_to_idx(length))
        
        # Add the beam
        height_field[beam_start_x:beam_end_x, min_y:max_y] = beam_height

        # Place goal near beam end, at beam center
        goal_x = beam_end_x - m_to_idx(beam_length/4)
        goals[i+1] = [goal_x, (min_y+max_y)//2]

        # Move forward for next beam, add gap
        cur_x = beam_end_x + m_to_idx(gap_length)

    # Fill in last part with flat area for final goal
    final_x = min(cur_x, m_to_idx(length)-1)
    height_field[final_x:, :] = 0
    goals[-1] = [min(final_x + m_to_idx(0.3), m_to_idx(length)-1), mid_y] 

    return height_field, goals