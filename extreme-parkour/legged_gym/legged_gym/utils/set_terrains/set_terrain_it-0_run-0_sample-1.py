import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Urban-style parkour: series of low, wide railings and variable curbs for precision stepping and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # -- Parameters for "urban parkour" elements --
    # Difficulty controls rail (beam) height, curb height, and spacing
    min_beam_height = 0.04 + 0.04 * difficulty  # 4-8cm low rails
    max_beam_height = 0.06 + 0.10 * difficulty  # up to 16cm
    min_curb_height = 0.05 + 0.03 * difficulty  # 5-8cm curbs
    max_curb_height = 0.08 + 0.09 * difficulty  # up to 17cm
    beam_width = 0.12 + 0.04 * (1-difficulty)   # 12-16cm narrow "rails"
    curb_width = 0.40 + 0.20 * (1-difficulty)   # 40-60cm curbs
    beam_length = 1.2 + 2.5 * (1-difficulty)    # 1.2-3.7m for each railing
    curb_length = 0.7 + 1.1 * (1-difficulty)    # 0.7-1.8m for each curb

    min_spacing = 0.5
    max_spacing = 1.0 + 0.8 * difficulty

    mid_y = m_to_idx(width / 2)

    # -- Flat spawn area --
    spawn_length = m_to_idx(2)
    height_field[0:spawn_length, :] = 0
    goals[0] = [spawn_length-1, mid_y]

    cur_x = spawn_length

    # Helper for safe margin (nothing within 0.5m of course edges)
    safe_margin_idx = m_to_idx(0.5)
    y_min = safe_margin_idx
    y_max = m_to_idx(width) - safe_margin_idx

    # Helper for placing and storing obstacles
    def place_beam(start_x, length, center_y, width, height, goal_idx):
        start_idx = int(start_x)
        end_idx = int(np.clip(start_x + m_to_idx(length), 0, height_field.shape[0]))
        half_w = m_to_idx(width / 2)
        y1 = int(np.clip(center_y - half_w, safe_margin_idx, height_field.shape[1] - safe_margin_idx))
        y2 = int(np.clip(center_y + half_w, safe_margin_idx, height_field.shape[1] - safe_margin_idx))
        height_field[start_idx:end_idx, y1:y2] = height
        # Goal at middle of the beam
        goals[goal_idx] = [start_idx + (end_idx - start_idx)//2, (y1 + y2)//2]

    def place_curb(start_x, length, center_y, width, height, goal_idx):
        # Like beam but wider: curb
        place_beam(start_x, length, center_y, width, height, goal_idx)  # alias

    # Alternate beams (narrow rails) and wide curbs in a zig-zag
    for i in range(1, 8):
        if i % 2 == 1:
            # Beam ("rail")
            b_len = beam_length + random.uniform(-0.2, 0.2)
            b_ht = random.uniform(min_beam_height, max_beam_height)
            centery = int(np.clip(mid_y + random.randint(-m_to_idx(1.0), m_to_idx(1.0)), y_min, y_max))
            place_beam(cur_x, b_len, centery, beam_width, b_ht, i)
            cur_x += m_to_idx(b_len)
        else:
            # Curb step (wide/low)
            c_len = curb_length + random.uniform(-0.1, 0.1)
            c_ht = random.uniform(min_curb_height, max_curb_height)
            centery = int(np.clip(mid_y + random.randint(-m_to_idx(1.1), m_to_idx(1.1)), y_min, y_max))
            place_curb(cur_x, c_len, centery, curb_width, c_ht, i)
            cur_x += m_to_idx(c_len)

        # Add gap ("jump"/step down to ground)
        if i < 7:
            gap_size = random.uniform(min_spacing, max_spacing)
            gap_idx = m_to_idx(gap_size)
            cur_x += gap_idx  # gap = floor (height 0)

    # Fill remainder with flat ground to finish
    cur_x = int(np.clip(cur_x, 0, height_field.shape[0]))
    height_field[cur_x:, :] = 0

    return height_field, goals