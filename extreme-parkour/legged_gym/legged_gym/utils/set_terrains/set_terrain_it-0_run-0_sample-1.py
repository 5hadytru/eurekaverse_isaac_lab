import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of staggered 'stepping stone' blocks (broad but slightly offset), above pits, for high-precision foot placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- Staggered wide "stepping stones" above a pit ---
    # Platform (block) params
    stone_length = 0.7 + 0.35 * (1-difficulty)    # longer at low difficulty, shorter at high
    stone_length_idx = m_to_idx(stone_length)
    stone_width = 1.0  # always at least 1 meter
    stone_width_idx = m_to_idx(stone_width)
    pit_depth = 0.5 + 0.6 * difficulty            # deep pit at higher difficulty
    block_height = 0.10 + 0.25 * difficulty       # higher at larger difficulty

    pit_start_x = m_to_idx(2)    # spawn area flat
    spawn_length = m_to_idx(2)
    field_x, field_y = height_field.shape

    # Set spawn area (flat)
    height_field[:pit_start_x, :] = 0

    # Set pit region after spawn area (robot falls if misses a block)
    height_field[pit_start_x:, :] = -pit_depth

    # Stagger blocks right and left;
    stones = 6
    min_stone_gap = 0.25 + 0.3 * difficulty  # meters; stones further apart as difficulty increases
    max_stone_gap = 0.4 + 0.5 * difficulty
    min_offset = 0.2
    max_offset = 0.5 + 0.8 * difficulty      # y-offset is harder at higher diff

    center_y = m_to_idx(width / 2)
    cur_x = pit_start_x

    # First goal: at the first spawn area
    goals[0] = [m_to_idx(1), center_y]

    # Helper to add a block platform
    def add_block(center_x, center_y, length_idx, width_idx, height):
        x1 = int(center_x - length_idx // 2)
        x2 = int(center_x + length_idx // 2)
        y1 = int(center_y - width_idx // 2)
        y2 = int(center_y + width_idx // 2)
        # Clip indices within height_field
        x1 = max(x1, 0); x2 = min(x2, field_x)
        y1 = max(y1, 0); y2 = min(y2, field_y)
        height_field[x1:x2, y1:y2] = height

    last_stone_y = center_y
    direction = 1  # start by offsetting right

    for i in range(stones):
        # X position: current
        x = int(cur_x + stone_length_idx//2)
        # Y offset: zigzag
        y_offset = m_to_idx(random.uniform(min_offset, max_offset) * direction)
        stone_center_y = np.clip(center_y + y_offset, m_to_idx(0.6), m_to_idx(width-0.6))
        add_block(x, stone_center_y, stone_length_idx, stone_width_idx, block_height)
        # Place a goal just over the center of the block
        goals[i+1] = [x, stone_center_y]
        last_stone_y = stone_center_y

        # Prepare for next block
        gap = m_to_idx(random.uniform(min_stone_gap, max_stone_gap))
        cur_x += stone_length_idx + gap
        direction *= -1  # alternate left and right

        # Ensure block doesn't run out of field
        if cur_x + stone_length_idx >= field_x-m_to_idx(1):
            cur_x = field_x-m_to_idx(1.5*stone_length)  # pack last block in if too close to the end
            break

    # Last (8th) goal: a flat "exit pad" after the last stepping stone
    exit_pad_x = int(min(cur_x + m_to_idx(0.7), field_x-1))
    exit_pad_w = stone_width_idx
    exit_pad_h = 0
    y1 = int(last_stone_y - exit_pad_w//2)
    y2 = int(last_stone_y + exit_pad_w//2)
    # Fill the ground up after last stone
    height_field[exit_pad_x:, y1:y2] = exit_pad_h
    goals[7] = [exit_pad_x, last_stone_y]

    # Fill any unused goals with the last valid goal
    for i in range(stones+2, 8):
        goals[i] = goals[7]

    return height_field, goals