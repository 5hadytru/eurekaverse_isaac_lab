import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping Stones: A series of floating rectangular 'stones' in a shallow water trough, testing precise stepping."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((4, 2))

    # --- Terrain Plan ---
    # The course is a shallow sunken trough, with regularly spaced elevated "stones" forming a stepping path.
    # Each obstacle consists of one stone, spaced so the quadruped must place its feet precisely to cross.
    # The difficulty increases by making the stones smaller and more widely spaced,
    # and by introducing mild lateral (sideways) offset at high difficulty.
    # The water trough (the negative-height region) encourages careful placement: stepping off stones is "undesirable".
    # Robot must traverse a gentle zigzag across stones to each goal.

    # --- Parameters dependent on difficulty ---
    trough_depth = -0.15 - 0.25 * difficulty    # Depth of trough
    stone_height = 0.0                          # Height of stones, at ground level
    num_stones = 4                              # Number of stones/goals

    # Stone size shrinks with difficulty:
    min_stone_len, max_stone_len = 0.55, 1.0
    min_stone_wid, max_stone_wid = 0.5, 1.2
    stone_length = np.linspace(max_stone_len, min_stone_len, num_stones) - 0.35 * difficulty
    stone_width = np.linspace(max_stone_wid, min_stone_wid, num_stones) - 0.4 * difficulty
    stone_length = np.clip(stone_length, 0.45, None)
    stone_width = np.clip(stone_width, 0.4, None)

    # Stone spacing increases as difficulty increases
    space0 = 1.5   # meters between spawn and first stone
    space_stone = np.linspace(1.1, 2.0, num_stones-1) + 1.0 * difficulty

    # Lateral offset for some stones at high difficulty
    y_center = m_to_idx(width / 2)
    y_amplitudes = np.zeros(num_stones)
    if difficulty > 0.6:
        y_amplitudes = np.array([0.0, 0.5, -0.4, 0.0]) * (difficulty-0.5) * 1.8

    trough_start_x = m_to_idx(2)   # Allow 2m for spawn area
    height_field[0:trough_start_x, :] = 0.        # Flat spawn region

    # Set entire region after spawn as sunken trough (simulated water)
    height_field[trough_start_x:, :] = trough_depth

    # Place stones and goals
    cur_x = trough_start_x
    for i in range(num_stones):
        # Stone center positions
        if i == 0:
            # First stone is further after spawn
            x = m_to_idx(2 + space0)
        else:
            x = cur_x + m_to_idx(space_stone[i-1])

        # Lateral offset for zigzag (smaller offset at low difficulty)
        y = y_center + m_to_idx(y_amplitudes[i])

        # Stone dimensions
        l = m_to_idx(stone_length[i])
        w = m_to_idx(stone_width[i])

        # Stone boundaries; ensure within bounds
        x1 = int(max(x - l//2, trough_start_x))
        x2 = int(min(x + l//2, m_to_idx(length)))
        y1 = int(max(y - w//2, 0))
        y2 = int(min(y + w//2, m_to_idx(width)))

        # Place stone at height 0 (ground level)
        height_field[x1:x2, y1:y2] = stone_height

        # Set the goal at the center of this stone
        goals[i, :] = [ (x1 + x2)//2, (y1 + y2)//2 ]

        # Next stone is placed further down the course
        cur_x = x

    # Final region after last stone returns to flat ground (exit "island")
    x_end = int(min(cur_x + m_to_idx(1.0), m_to_idx(length)))
    height_field[x_end:, :] = 0.

    return height_field, goals