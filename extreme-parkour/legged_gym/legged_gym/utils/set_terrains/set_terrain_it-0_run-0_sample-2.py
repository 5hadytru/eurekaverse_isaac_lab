import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of stepping stones: narrow, widely spaced blocks emulating urban 'parkour bricks' across a shallow pit. Tests precise foot placement and jumping."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((4, 2))

    # --- Setup parameters ---
    spawn_length = m_to_idx(2.0)
    n_stepping_stones = 4
    mid_y = m_to_idx(width / 2)
    pit_height = -0.5  # How deep the 'pit' between stones is
    stone_height = 0.05 + 0.15 * difficulty  # Stone slightly raised from ground
    stone_length = 0.45 + 0.15 * difficulty  # Long axis of brick
    stone_width = 0.45 + 0.07 * (1-difficulty)  # Short axis; minimal width at high difficulty
    stone_gap = 0.5 + 1.05 * difficulty  # Distance between stones (meters)
    y_jitter_max = 0.35 if difficulty > 0.5 else 0.17  # Sideways offset for balance/steering, more at higher difficulty

    stone_length_idx = m_to_idx(stone_length)
    stone_width_idx = m_to_idx(stone_width)
    stone_gap_idx = m_to_idx(stone_gap)

    # --- Set pit ---
    height_field[spawn_length:, :] = pit_height

    # --- Clear spawn area ---
    height_field[:spawn_length, :] = 0
    goals[0] = [m_to_idx(1.0), mid_y]  # Start near spawn

    current_x = spawn_length + m_to_idx(0.4)  # First stone a short distance after spawn

    for i in range(n_stepping_stones):
        # Place stone with width along y direction and length along x
        y_jitter = 0
        if i > 0:  # First stone is central, later stones can be offset
            y_jitter = int(random.uniform(-y_jitter_max, y_jitter_max) / field_resolution)
        stone_mid_y = mid_y + y_jitter
        x1 = int(current_x - stone_length_idx // 2)
        x2 = int(x1 + stone_length_idx)
        y1 = int(stone_mid_y - stone_width_idx // 2)
        y2 = int(y1 + stone_width_idx)
        # Keep indices within bounds
        x1 = max(0, x1)
        x2 = min(m_to_idx(length), x2)
        y1 = max(0, y1)
        y2 = min(m_to_idx(width), y2)
        # Raise stone surface
        height_field[x1:x2, y1:y2] = stone_height
        # Set goal at middle of each stone
        goals[i] = [int((x1+x2)/2), int((y1+y2)/2)]
        current_x = x2 + stone_gap_idx

    # --- Ensure last part past last stone is traversable ---
    final_floor_start = min(m_to_idx(length), current_x - m_to_idx(0.2))
    height_field[final_floor_start:, :] = 0
    goals[-1] = [int((final_floor_start + m_to_idx(length)-1)//2), mid_y]  # Final goal after stepping stones

    return height_field, goals