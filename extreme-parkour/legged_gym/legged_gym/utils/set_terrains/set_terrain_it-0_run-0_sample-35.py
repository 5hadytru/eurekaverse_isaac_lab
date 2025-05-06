import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping-stone pillars: The robot must step, balance, and sometimes hop across a sequence of round, raised stepping-stone columns (pillars) of varying heights."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Pillar parameters
    # Pillar diameter (ensure > quadruped width, get harder with smaller size), but never less than 0.45m
    min_pillar_d = 0.45
    max_pillar_d = 0.9
    pillar_diameter = max(min_pillar_d, max_pillar_d - 0.5 * difficulty)
    pillar_radius_idx = m_to_idx(pillar_diameter / 2)

    # Pillar height
    min_height = 0.05 + (0.10 * difficulty)
    max_height = 0.12 + (0.20 * difficulty)

    # Gaps between pillars
    min_gap = 0.18 + (difficulty * 0.32)  # From easy step to wide hop
    max_gap = 0.35 + (difficulty * 0.45)
    n_pillars = 7

    # Pillar starting position (leave safe space at spawn)
    spawn_len_idx = m_to_idx(2.0)
    x_positions = [spawn_len_idx]
    y_center = m_to_idx(width / 2)

    # Pre-calculate pillar centers
    for i in range(n_pillars):
        if i == 0:
            x = spawn_len_idx + m_to_idx(0.6 + 0.1 * random.uniform(-1, 1))  # first pillar
        else:
            gap = random.uniform(min_gap, max_gap)
            x = x_positions[-1] + m_to_idx(gap + pillar_diameter)
        # Randomize y-position for challenge, but keep within bounds
        y_offset = int(m_to_idx( 0.2 + 0.25 * difficulty) * random.uniform(-1, 1))
        y = np.clip(y_center + y_offset, pillar_radius_idx, m_to_idx(width) - pillar_radius_idx - 1)
        x_positions.append(x)
        if i < 7:
            # Place the goal at the next pillar center.
            goals[i+1] = [x, y]
        # Place pillar
        pillar_height = random.uniform(min_height, max_height)
        x0, x1 = int(x - pillar_radius_idx), int(x + pillar_radius_idx)
        y0, y1 = int(y - pillar_radius_idx), int(y + pillar_radius_idx)
        for xi in range(x0, x1+1):
            for yi in range(y0, y1+1):
                # Round stepping stone: keep only those within the radius
                if ( (xi-x)**2 + (yi-y)**2 ) <= pillar_radius_idx**2:
                    height_field[xi, yi] = pillar_height

    # Set pit: everything except pillars is a pit of depth -1.1  (except spawn)
    height_field[spawn_len_idx:, :] = -1.1
    for i in range(n_pillars):
        x = x_positions[i+1]
        y = goals[i+1][1]
        x0, x1 = int(x - pillar_radius_idx), int(x + pillar_radius_idx)
        y0, y1 = int(y - pillar_radius_idx), int(y + pillar_radius_idx)
        for xi in range(x0, x1+1):
            for yi in range(y0, y1+1):
                if ( (xi-x)**2 + (yi-y)**2 ) <= pillar_radius_idx**2:
                    # overwrite pit with pillar
                    height_field[xi, yi] = height_field[xi, yi]

    # Set spawn area (start) to 0 height and easy walking
    height_field[:spawn_len_idx, :] = 0.0 
    # First goal: at easy launch point just before first pillar
    goals[0] = [spawn_len_idx - m_to_idx(0.5), y_center]

    # Final goal: after last pillar, on stable ground
    end_pad = m_to_idx(0.6)
    if x_positions[-1] + end_pad < m_to_idx(length) - 1:
        height_field[int(x_positions[-1]+1):, :] = 0.0
        goals[-1] = [x_positions[-1] + end_pad, y_center]
    else:
        goals[-1] = [m_to_idx(length)-2, y_center]

    return height_field, goals