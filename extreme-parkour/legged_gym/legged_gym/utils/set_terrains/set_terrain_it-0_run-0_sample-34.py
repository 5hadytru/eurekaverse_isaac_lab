import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone sequence: robot must precisely step over a series of narrow, offset stone blocks above a deep pit, testing lateral agility and careful foot placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Main parameters for the stepping stone course
    # Each "stone" is a rectangular raised platform, 0.5-0.7m long, 0.45-0.6m wide; spaced by 0.3-0.7m; alternates left and right
    stone_length = 0.5 + 0.2 * difficulty
    stone_width = 0.45 + 0.15 * (1-difficulty)  # More difficult = narrower
    stone_height = 0.1 + 0.25 * difficulty      # Harder= higher
    gap = 0.35 + 0.4 * difficulty               # Spacing between stones
    lateral_offset = 0.30 + 0.20 * difficulty   # How far off the center each is

    n_stones = 8  # One for each goal
    start_x = 2.0  # meters: start placing stones after robot spawn area
    course_length = (n_stones * stone_length) + ((n_stones-1) * gap)
    usable_length = length - start_x * 1.2      # leave a bit of exit space

    assert course_length <= usable_length, "Stepping stones do not fit in the field, adjust sizes!"

    # Pit parameters: everything outside a stone is -0.8m
    pit_height = -0.8

    # Fill post-spawn area with pit, then overwrite stepping stones as raised
    spawn_x_idx = m_to_idx(start_x)
    height_field[spawn_x_idx:, :] = pit_height

    # Place each stepping stone and corresponding goal
    # Stones alternate left/right of center line
    mid_y = m_to_idx(width / 2)
    stone_l = m_to_idx(stone_length)
    stone_w = m_to_idx(stone_width)
    gap_l = m_to_idx(gap)
    lateral_off_idx = m_to_idx(lateral_offset)
    between_stones_clearance = 0.18  # [m], buffer edge-to-edge so even at minimum width robot has margin

    cur_x = spawn_x_idx
    for i in range(n_stones):
        # Alternate left/right, bias is for i=0 -> center
        if i % 2 == 0:
            y_c = mid_y - lateral_off_idx
        else:
            y_c = mid_y + lateral_off_idx
        y1 = max(0, y_c - stone_w // 2)
        y2 = min(height_field.shape[1], y_c + (stone_w + 1) // 2)
        x1 = cur_x
        x2 = min(height_field.shape[0], cur_x + stone_l)
        height_field[x1:x2, y1:y2] = stone_height  # Place stone

        # Goal: center of stone
        goals[i] = [x1 + (stone_l // 2), y_c]

        # Step cur_x to next stone
        cur_x = x2 + gap_l

    # Flat safe exit after last stone
    exit_x1 = cur_x
    exit_x2 = min(m_to_idx(length), cur_x + m_to_idx(1.0))
    height_field[exit_x1:exit_x2, :] = 0.0

    # Set the first goal (spawn area) to center just *before* first stone for initial acquisition
    goals[0] = [m_to_idx(1.5), mid_y]

    # Ensure 8 valid goal positions, always in-bounds
    for i in range(8):
        goals[i, 0] = np.clip(goals[i, 0], 0, height_field.shape[0]-1)
        goals[i, 1] = np.clip(goals[i, 1], 0, height_field.shape[1]-1)

    return height_field, goals