import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of narrow balance beams over deep pits, testing dynamic balancing and precise foot placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        if isinstance(m, list) or isinstance(m, tuple):
            return [round(i / field_resolution) for i in m]
        return np.round(m / field_resolution).astype(np.int16)

    # Allocate the terrain field and goals array
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    FIELD_LEN = m_to_idx(length)
    FIELD_WID = m_to_idx(width)

    # Constants for obstacle sizes, relative to robot size (robot ~0.645 x 0.28 m)
    beam_width = 0.34 + 0.2*(1-difficulty)      # beams are 0.34m wide at hard, up to 0.54m at easy
    beam_width_idx = m_to_idx(beam_width)
    beam_height = 0.10 + 0.13*difficulty       # 10cm ~ 23cm up
    beam_gap = 0.66 + 1.05*difficulty          # 0.66m up to 1.7m gap between beams (forces jumps at higher diff)
    pit_depth = -0.38 - 0.62*difficulty        # pits are -0.38m to -1.0m deep

    beam_length = 1.85 + 1.2*(1-difficulty)    # 1.85m beam at hard, up to 3.05m at easy
    beam_length_idx = m_to_idx(beam_length)
    min_goal_offset = m_to_idx(0.25)           # keep goals well within beams

    y_center = FIELD_WID // 2
    beam_y0 = y_center - beam_width_idx // 2
    beam_y1 = beam_y0 + beam_width_idx

    # Flat spawn section
    spawn_len = m_to_idx(2)
    height_field[:spawn_len, :] = 0

    # Place first goal (start)
    goals[0] = [spawn_len - m_to_idx(0.5), y_center]

    # Lay out 6 balance beams separated by pits, last goal on flat ground
    cur_x = spawn_len

    for i in range(6):
        # Place pit (set to negative height)
        pit_start = cur_x
        pit_end = cur_x + m_to_idx(0.22 * (1-difficulty) + 0.12) if i > 0 else cur_x # skip at first (already flat)
        if i > 0:
            pit_width = pit_end - pit_start
            height_field[pit_start:pit_end, :] = pit_depth

        cur_x = pit_end
        # Add small lateral offset at harder difficulties for extra balancing challenge
        y_offset = int((random.random()-0.5) * (FIELD_WID // (6 + 6*(1-difficulty))))
        b_y0 = np.clip(beam_y0 + y_offset, 0, FIELD_WID-beam_width_idx)
        b_y1 = b_y0 + beam_width_idx

        # Place the beam
        beam_start = cur_x
        beam_end = cur_x + beam_length_idx
        beam_end = min(beam_end, FIELD_LEN)  # prevent overrun at end
        height_field[beam_start:beam_end, b_y0:b_y1] = beam_height
        # Set pit everywhere else (other than beam)
        height_field[beam_start:beam_end, :b_y0] = pit_depth
        height_field[beam_start:beam_end, b_y1:] = pit_depth

        # Set goal at center of beam
        mid_x = (beam_start + beam_end) // 2
        mid_y = (b_y0 + b_y1) // 2
        goals[i+1] = [mid_x, mid_y]

        # Step forward to next pit
        cur_x = beam_end + m_to_idx(beam_gap)
        if cur_x >= FIELD_LEN - m_to_idx(1):
            break  # finish before running off the area

    # Final flat section (safe zone)
    safe_zone_st = min(cur_x, FIELD_LEN)
    height_field[safe_zone_st:, :] = 0
    goals[7] = [safe_zone_st + m_to_idx(0.5), y_center]

    # Fill any leftover unset goals (if finished early)
    for j in range(8):
        goals[j, 0] = np.clip(goals[j, 0], 0, FIELD_LEN-1)
        goals[j, 1] = np.clip(goals[j, 1], 0, FIELD_WID-1)
    return height_field, goals