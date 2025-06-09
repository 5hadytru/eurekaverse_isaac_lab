import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping-stone sequence: A series of narrow, alternating-offset flat pads over a deep trench to challenge precise foot placement and lateral agility."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # Pad and gap configuration (stepping stones over a trench)
    pad_length = 0.5 + 0.2 * (1-difficulty)      # pads get shorter at higher difficulty
    pad_width = 0.5 + 0.4 * (1-difficulty)       # pads get narrower at higher difficulty
    pad_height = 0.15 + 0.10 * difficulty        # pads become higher with difficulty
    gap_length = 0.40 + 0.45 * difficulty        # gaps get longer at higher difficulty

    pad_length_idx = m_to_idx(pad_length)
    pad_width_idx = m_to_idx(pad_width)
    pad_height = float(pad_height)
    gap_length_idx = m_to_idx(gap_length)

    # Terrain bounds
    total_x = m_to_idx(length)
    total_y = m_to_idx(width)

    # Create a central trench
    trench_depth = -1.1
    height_field[:, :] = trench_depth

    # Spawn area: flat ground for spawning
    safe_zone_idx = m_to_idx(2)
    height_field[0:safe_zone_idx, :] = 0.0

    # Place the stepping stones in a zig-zag manner
    cur_x = safe_zone_idx
    n_pads = 4
    lateral_shifts = [0.0, 0.8, -0.9, 0.7]  # meters offset from center, zig-zag

    for i in range(n_pads):
        # Pad center
        # Make sure pads aren't too close to the edge
        pad_offset_y = lateral_shifts[i % len(lateral_shifts)] * (1 + 0.2*random.uniform(-1,1)*difficulty)
        pad_center_y = (width / 2) + pad_offset_y
        pad_center_y = max(pad_width/2 + 0.05, min(width - pad_width/2 - 0.05, pad_center_y))
        # Indices
        x1 = cur_x
        x2 = min(total_x, x1 + pad_length_idx)
        center_y_idx = m_to_idx(pad_center_y)
        half_pw = pad_width_idx//2
        y1 = max(0, center_y_idx - half_pw)
        y2 = min(total_y, center_y_idx + half_pw)
        # Place pad
        height_field[x1:x2, y1:y2] = pad_height
        # Set goal in pad center
        pad_mid_x = (x1+x2)//2
        goals[i] = [pad_mid_x, center_y_idx]
        # Advance to next pad
        cur_x = x2 + gap_length_idx

    # Final area: safe/flat finish pad
    finish_pad_start = min(cur_x, total_x - m_to_idx(1))
    finish_pad_end = total_x
    height_field[finish_pad_start:finish_pad_end, :] = 0.0
    goals[4] = [ (finish_pad_start+finish_pad_end)//2, m_to_idx(width/2) ]

    # First goal is always the starting area
    goals[0] = [ m_to_idx(1), m_to_idx(width/2) ]

    return height_field, goals