import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone course: A sequence of narrow, tall, spaced-apart pillars for precise jumping."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    mid_y = m_to_idx(width) // 2

    # Pillar parameters (stepping stones)
    # Pillars force precise jumping, accuracy, and balance
    num_pillars = 7  # one goal on each pillar, last goal beyond the last pillar

    # Pillar size: narrow, but not less than 0.45m x 0.45m (still room for quadruped, but challenging)
    min_pillar_size = 0.45
    max_pillar_size = 0.6 - 0.15 * difficulty  # smaller at higher difficulty
    min_pillar_size_idx = m_to_idx(min_pillar_size)

    # Height of pillars: ranging from 0.18m (easy) up to 0.45m (hard)
    pillar_height_min = 0.18 + 0.22 * difficulty
    pillar_height_max = 0.25 + 0.40 * difficulty

    # Gaps between pillars: wider at higher difficulty (0.35m up to 0.9m)
    min_gap = 0.35 + 0.5 * difficulty
    max_gap = 0.45 + 0.7 * difficulty

    # Clear spawn region (always flat)
    spawn_len = m_to_idx(2.0)
    height_field[:spawn_len, :] = 0
    # Goal 0: set at end of spawn region
    goals[0] = [spawn_len - m_to_idx(0.5), mid_y]

    # Set the rest of the field to a deep pit
    height_field[spawn_len:, :] = -1.2  # Pit depth, untraversable

    cur_x = spawn_len

    # Helper: place a square pillar in the field, centered at (x, y)
    def place_pillar(center_x, center_y, size, height):
        half = size // 2
        x_start = max(int(center_x - half), 0)
        x_end   = min(int(center_x + half + 1), height_field.shape[0])
        y_start = max(int(center_y - half), 0)
        y_end   = min(int(center_y + half + 1), height_field.shape[1])
        height_field[x_start:x_end, y_start:y_end] = height

    # Parameters to wiggle the y positions: small zig-zag, but always leaves full pillar inside bounds
    min_pillar_margin = m_to_idx(0.25)
    y_lower = min_pillar_margin
    y_upper = m_to_idx(width) - min_pillar_margin
    # We'll move each pillar up/down by at most 0.7m from center, but not off the course
    y_jitter = int(m_to_idx(0.7))

    random.seed(0)  # ensure reproducibility of course
    for i in range(num_pillars):
        # Set inter-pillar gap and pillar position
        gap = random.uniform(min_gap, max_gap)
        gap_idx = m_to_idx(gap)

        # Pillar size and height
        pillar_size = random.uniform(min_pillar_size, max_pillar_size)
        pillar_size_idx = m_to_idx(pillar_size)
        pillar_height = random.uniform(pillar_height_min, pillar_height_max)

        # Pillar y position (allow gentle zig-zag)
        if i == 0:
            pillar_y = mid_y
        else:
            # Try to shift by -y_jitter, 0, or +y_jitter
            direction = random.choice([-1, 0, 1])
            step = direction * random.randint(0, y_jitter)
            pillar_y = goals[i][1].astype(int) + step
            # Keep within margin
            pillar_y = max(y_lower, min(y_upper, pillar_y))

        # Pillar x position
        pillar_x = int(cur_x + gap_idx + pillar_size_idx // 2)

        # Place the pillar
        place_pillar(pillar_x, pillar_y, pillar_size_idx, pillar_height)

        # Update cur_x to the end of this pillar
        cur_x = pillar_x + pillar_size_idx // 2

        # Place a goal in the center of the pillar's top
        goals[i+1] = [pillar_x, pillar_y]

    # Final goal is on the ground, 1m after last pillar and centered
    last_goal_x = min(cur_x + m_to_idx(1.0), m_to_idx(length) - 4)
    height_field[last_goal_x:, :] = 0  # flat ground after the course
    goals[7] = [last_goal_x, mid_y]

    return height_field, goals