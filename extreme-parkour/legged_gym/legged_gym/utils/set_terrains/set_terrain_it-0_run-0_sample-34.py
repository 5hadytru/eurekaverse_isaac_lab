import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping-stone 'urban blocks' course: jumping/walking atop sequential narrow rectangular blocks, testing balance and narrow foothold navigation."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- Parameters ---
    # Block and gap sizes
    block_width = 0.5 + 0.2 * (1 - difficulty)    # 0.7m at easy, 0.5m at hard
    block_length = 1.2 - 0.3 * difficulty         # 1.2m at easy, 0.9m at hard (always > robot length)
    gap_min = 0.15 + 0.20 * difficulty            # minimum 0.15m gap at easy, up to 0.35m+ at hard
    gap_max = gap_min + 0.2 * difficulty          # more variability with difficulty

    block_height = 0.15 + 0.23 * difficulty       # Easy: 0.15m, Hard: 0.38m (max ~robot's knee)
    pit_depth = 0.60 + 0.4 * difficulty           # Deep pits

    # Convert to indices
    block_width_i = m_to_idx(block_width)
    block_length_i = m_to_idx(block_length)
    gap_min_i, gap_max_i = m_to_idx(gap_min), m_to_idx(gap_max)
    block_height = float(block_height)
    pit_height = -float(pit_depth)

    margin_y = m_to_idx(0.5)   # keep all blocks away from edge of field

    # --- Place initial flat ground for spawn area ---
    spawn_length = m_to_idx(2)
    height_field[:spawn_length, :] = 0
    mid_y = m_to_idx(width / 2)

    # --- Construction loop ---
    cur_x = spawn_length
    # Stride pattern: most blocks straight, sometimes slight left/right offset
    y_positions = [mid_y]
    n_blocks = 7

    # First goal is in spawn region
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    for i in range(n_blocks):
        # Slight y offset: occasionally make the robot shift left/right, but not edge to edge
        if i > 0:
            if random.random() < 0.45:
                dy = m_to_idx(random.choice([-0.5, 0.5]))
                new_y = min(max(y_positions[-1] + dy,
                                margin_y + block_width_i//2),
                            m_to_idx(width) - margin_y - block_width_i//2)
                y_positions.append(int(new_y))
            else:
                y_positions.append(y_positions[-1])

        y_c = y_positions[i]
        x1 = cur_x
        x2 = min(x1 + block_length_i, m_to_idx(length) - 1)

        # Carve a pit around the block first
        pit_margin = m_to_idx(0.05)
        pit_x1 = max(x1 - pit_margin, spawn_length)
        pit_x2 = min(x2 + pit_margin, m_to_idx(length) - 1)
        pit_y1 = max(y_c - block_width_i//2 - pit_margin, 0)
        pit_y2 = min(y_c + block_width_i//2 + pit_margin, m_to_idx(width) - 1)
        height_field[pit_x1:pit_x2, pit_y1:pit_y2] = pit_height

        # Add the block
        b_y1 = int(y_c - block_width_i // 2)
        b_y2 = int(y_c + (block_width_i + 1) // 2)
        height_field[x1:x2, b_y1:b_y2] = block_height

        # Place a goal approximately at the center of this block
        goals[i+1] = [int((x1 + x2) // 2), int((b_y1 + b_y2) // 2)]

        # Next block: advance x by block + random gap
        gap = random.randint(gap_min_i, gap_max_i)
        cur_x += block_length_i + gap

        # Avoid placing blocks beyond the field
        if cur_x + block_length_i >= m_to_idx(length) - 1:
            break

    # Final goal: flat finishing area at exit
    fin_margin = m_to_idx(1)
    finish_x1 = min(cur_x, m_to_idx(length) - fin_margin)
    height_field[finish_x1:, :] = 0
    goals[7] = [min(finish_x1 + m_to_idx(0.5), m_to_idx(length) - 1), mid_y]

    return height_field, goals