import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Eight step obstacles ("stepping stones") across a deep trench requiring precise, sequential stepping."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Stepping stone/trench parameters
    # Trench is a pit that occupies the center of the course; stones are raised columns above the pit
    trench_start_x = m_to_idx(2)  # Ensure the spawn area is clear
    trench_end_x = m_to_idx(length - 1)  # Leave 1m at far end for flat goal
    trench_width = m_to_idx(width * (0.45 + 0.25 * difficulty))  # Trench fills 45%-70% of course width with difficulty

    pit_depth = -0.85 - 0.4 * difficulty  # Pit is deeper at high difficulty
    # Set trench to negative height (pit)
    trench_y_mid = m_to_idx(width // 2)
    trench_y1 = (m_to_idx(width) - trench_width) // 2
    trench_y2 = trench_y1 + trench_width
    height_field[trench_start_x:trench_end_x, trench_y1:trench_y2] = pit_depth

    # Place 8 stepping stones along a zigzag path across the trench
    # Stepping stone parameters
    stone_size = 0.5 + 0.25 * (1 - difficulty)  # Stones smaller with difficulty (min 0.5m, max 0.75m wide)
    stone_h = 0.18 + 0.15 * difficulty  # Stone heights are higher at high difficulty
    stone_dist_min = 1.0 - 0.3 * difficulty  # Closest stones at hardest setting

    stones_x = []
    stones_y = []

    trench_len = trench_end_x - trench_start_x
    step_count = 8

    spawn_area_x = m_to_idx(0.8)
    # Keep area before trench flat and safe for spawn
    height_field[0:trench_start_x, :] = 0

    # Zigzag stones: alternate left/right with some randomness, but stones always stay above pit
    for i in range(step_count):
        # Evenly spaced steps along x
        frac_x = (i + 1) / (step_count + 1)
        stone_x = trench_start_x + int(frac_x * trench_len)
        # Y position: zigzag left/right, but over trench
        zigzag_offset = ((-1) ** i) * (trench_width // 4 - m_to_idx(0.15 + 0.1 * random.random()))
        stone_y = (trench_y1 + trench_y2) // 2 + zigzag_offset

        # Clamp to trench bounds
        stone_y = max(trench_y1 + m_to_idx(stone_size//2+0.1), min(stone_y, trench_y2 - m_to_idx(stone_size//2+0.1)))

        # Place stone (square/circular area)
        x1 = max(0, stone_x - m_to_idx(stone_size / 2))
        x2 = min(height_field.shape[0], stone_x + m_to_idx(stone_size / 2))
        y1 = max(0, stone_y - m_to_idx(stone_size / 2))
        y2 = min(height_field.shape[1], stone_y + m_to_idx(stone_size / 2))
        height_field[x1:x2, y1:y2] = stone_h

        stones_x.append(stone_x)
        stones_y.append(stone_y)

        # Set goals at centers of stones
        goals[i] = [stone_x, stone_y]

    # First goal is close to spawn, on flat ground before pit
    goals[0] = [m_to_idx(1), m_to_idx(width) // 2]
    # Shift stones+goals forward so goal indices are always ascending
    for i in range(1, step_count):
        goals[i] = [stones_x[i], stones_y[i]]

    # Final (8th) goal: after the last stone, on safe ground beyond trench
    safe_final_x = trench_end_x + m_to_idx(0.5)
    safe_final_y = m_to_idx(width) // 2
    goals[7] = [min(safe_final_x, height_field.shape[0]-1), safe_final_y]
    # Make area behind trench flat and solid
    height_field[trench_end_x:, :] = 0

    # For visual convenience: fill corners and side edges of height_field 
    # with flat ground, so missed steps are clear and penalizing
    height_field[:, :trench_y1] = 0
    height_field[:, trench_y2:] = 0

    return height_field, goals