import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of 'stepping stone' narrow beams requiring careful walking and precise turning across a zig-zag gap."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for beams/gaps, scaling with difficulty
    min_beam_length = 1.2 - 0.4 * difficulty  # in meters
    max_beam_length = 1.8 - 0.7 * difficulty  # in meters
    beam_width = 0.45 - 0.1 * difficulty      # in meters, but never less than 0.4
    beam_width = max(0.4, beam_width)
    beam_height = 0.11 + 0.13 * difficulty    # small step up
    gap_length = 0.24 + 0.56 * difficulty     # meters
    pit_depth = -1.2                          # deep enough to force failure if fallen in
    step = 1                                  # advance goal index

    mid_y = m_to_idx(width) // 2
    spawn_length = m_to_idx(2)
    length_idx = m_to_idx(length)
    width_idx = m_to_idx(width)

    # 1. Flat spawn area
    height_field[:spawn_length, :] = 0
    
    # First goal: start straight ahead from spawn
    goals[0] = [m_to_idx(1.0), mid_y]

    # 2. Set all post-spawn region as pit except for beams
    height_field[spawn_length:, :] = pit_depth

    # 3. Generate beams in a zig-zag pattern
    cur_x = spawn_length              # current x index
    left_y = m_to_idx(1.10)
    right_y = width_idx - m_to_idx(1.10)
    center_y = mid_y

    directions = [0, 1, 0, -1, 0, 1, 0]  # sequence of direction deltas for zig and zag effect

    for i in range(7):  # Up to 7 beam segments (7 steps = 8 goals)
        # beam direction: even = horizontal, odd = turn left/right
        if i % 2 == 0:   # Centered horizontal beam
            beam_centre_y = center_y
        elif i % 4 == 1:
            beam_centre_y = left_y
        else:
            beam_centre_y = right_y

        # Vary beam length and possible y position with some randomness
        l = m_to_idx(np.random.uniform(min_beam_length, max_beam_length))
        w = m_to_idx(beam_width)
        y1 = beam_centre_y - w // 2
        y2 = beam_centre_y + w // 2

        # Beam position (x slice)
        x1 = cur_x
        x2 = min(cur_x + l, length_idx)

        # Place the beam (flat top, at height)
        height_field[x1:x2, y1:y2] = beam_height

        # Set the next goal at middle of this beam segment (to force robot to walk precisely)
        goal_x = (x1 + x2) // 2
        goal_y = (y1 + y2) // 2
        goals[step] = [goal_x, goal_y]
        step += 1

        cur_x = x2

        # Insert gap after each beam except last
        if step < 8:
            g = m_to_idx(gap_length)
            cur_x = int(cur_x + g)

    # 4. After last beam, fill remaining ground so robot does not fall
    height_field[cur_x:, :] = 0
    if step < 8:
        # Fill remaining goals down straight middle, if for some reason not 8
        for extra in range(step, 8):
            goals[extra] = [min(cur_x + m_to_idx(0.7), length_idx - 1), mid_y]

    return height_field, goals