import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone path: Repeated narrow raised pads and gaps for precise foot placement and jumping."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- Stepping Stone Parameters ---
    #
    # The stepping stone pads are like large paving stones: flat, slightly raised pads with gaps between them.
    # The pads get a little smaller and the gaps a little bigger with increased difficulty.
    #
    # All pads extend across most of the width of the course, but with random lateral offsets to force steering.
    # The quadruped must step onto these single-passage pads (avoiding the 'water' -- negative height gaps).
    #
    # At high difficulty, stones are just wide enough for the robot to land with little tolerance.
    #

    # Parameters in meters
    pad_length = 0.7 - 0.15 * difficulty    # Stepping stone (along x)
    pad_width_min = 0.7 - 0.25 * difficulty # Narrowness (along y)
    pad_width_max = 1.2 - 0.35 * difficulty
    gap_length = 0.37 + 0.43 * difficulty   # Distance between stones (x)
    gap_depth = -0.16 - 0.24 * difficulty   # Height of negative space between stones

    # For lateral displacement
    lateral_max_offset = 1.0 - 0.75 * difficulty  # Max offset reduces on hard (forces more straight line at hardest)
    pad_overlap = 0.18 + (0.1 * (1-difficulty))   # How much "lead-on" the pad gives in y to allow steering transitions

    # Spawn and goal logic
    pad_count = 6
    spawn_length = 2.0   # meters before first pad must not have obstacles
    cur_x = m_to_idx(spawn_length)

    # Place initial area (the spawn zone) as plain ground
    height_field[:cur_x, :] = 0
    y_center_idx = m_to_idx(width/2)

    # Place first goal at spawn position
    goals[0] = [m_to_idx(1.0), y_center_idx]

    # Helper function to add a pad at (start_x, y_center)
    def add_pad(start_x, y_center, pad_len_idx, pad_wid_idx, stone_height):
        y1 = max(0, y_center - pad_wid_idx//2)
        y2 = min(height_field.shape[1], y_center + pad_wid_idx//2)
        x1 = start_x
        x2 = min(height_field.shape[0], start_x + pad_len_idx)
        height_field[x1:x2, y1:y2] = stone_height

    # All gaps between stones are negative
    height_field[cur_x:, :] = gap_depth

    stone_height = 0.06 + 0.18*difficulty  # A little step up at easy, moderate at hard

    lateral_centers = []
    for i in range(pad_count):
        # Keep pads well in bounds, avoiding too close to edges
        pad_width = random.uniform(pad_width_min, pad_width_max)
        pad_wid_idx = m_to_idx(pad_width)
        pad_len_idx = m_to_idx(pad_length)

        # Pick center y coordinate randomly, avoiding too close to left/right side
        left_bound = m_to_idx(0.7) + pad_wid_idx//2
        right_bound = height_field.shape[1] - m_to_idx(0.7) - pad_wid_idx//2
        # Allow up to lateral_max_offset in y
        if i == 0:
            y_center = y_center_idx  # Start straight
        else:
            max_offset = m_to_idx(lateral_max_offset)
            prev_center = lateral_centers[-1]
            # Pick new center within offset limits, and course bounds
            y_center = np.clip(prev_center + random.randint(-max_offset, max_offset), left_bound, right_bound)
        lateral_centers.append(y_center)

        # Place pad
        add_pad(cur_x, y_center, pad_len_idx, pad_wid_idx, stone_height)

        # Place goal in the middle of this pad
        pad_center_x = cur_x + pad_len_idx//2
        goals[i+1] = [pad_center_x, y_center]

        # Move to next stone, with randomized gap for variety
        gap = gap_length + random.uniform(-0.05, 0.07)*difficulty
        cur_x += pad_len_idx + m_to_idx(gap)

    # Final pad ("dry land") to finish
    end_pad_x = min(cur_x, height_field.shape[0] - m_to_idx(0.9))
    final_pad_len = m_to_idx(1.1)
    final_pad_width = m_to_idx(1.3)
    add_pad(end_pad_x, y_center_idx, final_pad_len, final_pad_width, stone_height)
    goals[7] = [end_pad_x + final_pad_len//2, y_center_idx]

    # Ensure the rest of the course after last pad is regular ground (not negative)
    height_field[end_pad_x+final_pad_len:, :] = 0

    return height_field, goals