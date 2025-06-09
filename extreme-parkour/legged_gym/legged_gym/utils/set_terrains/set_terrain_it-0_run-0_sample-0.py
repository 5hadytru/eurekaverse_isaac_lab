import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone pillars: The robot crosses a series of narrow, tall pillars (stepping stones) above a deep pit, requiring precise foot placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Initialize the height field to flat ground
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # Parameters
    n_pillars = 5
    # Pillar width (from 0.4 - 0.6 m, wider at lower difficulty)
    pillar_width = 0.6 - 0.2 * difficulty  
    pillar_width_idx = max(m_to_idx(pillar_width), m_to_idx(0.4))
    # Pillar height (from 0.1 at low diff to 0.35 at high diff)
    h_min = 0.10 + 0.10 * difficulty
    h_max = 0.18 + 0.17 * difficulty

    # Between 1.8m and 2.2m gap between pillar centers
    gap_mu = 2.0 + 0.5*(difficulty - 0.5)  # range from 1.75 to 2.25
    gap_sigma = 0.10 + 0.20 * difficulty

    mid_y = m_to_idx(width) // 2

    # Make a deep pit after the spawn area (all cells set to negative height)
    spawn_length = m_to_idx(2)
    height_field[0:spawn_length,:] = 0.0         # Start area stays flat ground (spawn)
    pit_x_start = spawn_length
    height_field[pit_x_start:,:] = -1.1          # Pit is -1.1m deep

    # Place first goal at the front of the spawn platform
    goals[0] = [spawn_length-m_to_idx(0.5), mid_y]

    # Randomly offset the pillar row in y direction up to ±0.8m
    pillar_lane_offset = random.randint(-m_to_idx(0.8), m_to_idx(0.8))

    # Pillar placement
    pillar_x = []
    pillar_y = []
    cur_x = pit_x_start + m_to_idx(0.5)  # First pillar, some gap after spawn
    for i in range(n_pillars):
        # For each pillar, randomize y location (within ±0.7m) from centerline, and randomize pillar height
        if i == 0:
            center_y = mid_y + pillar_lane_offset
        else:
            sideways_offset = random.randint(-m_to_idx(0.7), m_to_idx(0.7))
            center_y = mid_y + sideways_offset + pillar_lane_offset

        h = np.random.uniform(h_min, h_max)
        x1 = max(cur_x - pillar_width_idx//2, 0)
        x2 = min(cur_x + pillar_width_idx//2, m_to_idx(length))
        y1 = max(center_y - pillar_width_idx//2, 0)
        y2 = min(center_y + pillar_width_idx//2, m_to_idx(width))

        # Draw pillar (raise above pit to its assigned height)
        height_field[x1:x2, y1:y2] = h

        # Place goal at the center of this pillar
        goals[i] = [ (x1 + x2) // 2, (y1 + y2) // 2 ]

        pillar_x.append((x1 + x2) // 2)
        pillar_y.append((y1 + y2) // 2)

        # Advance x location for next pillar, add random (per difficulty) gap
        gap = np.random.normal(gap_mu, gap_sigma)
        cur_x += m_to_idx(gap)

    # Ensure the last bit of the terrain is set above the pit to allow safe exit
    end_x = min(m_to_idx(length), int(cur_x) + m_to_idx(1.0))
    height_field[end_x:, :] = 0.0    # Raise the exit area to flat ground

    # Place final goal (5th) at the flat exit
    goals[4] = [end_x + m_to_idx(0.5), mid_y]

    # Make sure all goal indices are within bounds
    for i in range(5):
        goals[i,0] = np.clip(goals[i,0], 0, m_to_idx(length)-1)
        goals[i,1] = np.clip(goals[i,1], 0, m_to_idx(width)-1)

    return height_field, goals