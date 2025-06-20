import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone path: a sequence of narrow flat 'stepping stones' over a sunken alley tests precise foot placement and balancing."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # Parameters
    num_stones = 5           # Number of stones == number of goals (one per stone)
    pit_depth = -0.8 - 0.7 * difficulty      # Alley is always below starting ground, deeper at higher difficulty
    alley_start = m_to_idx(2.0)
    alley_end = m_to_idx(length - 1.0)
    mid_y = m_to_idx(width / 2)
    stone_width = max(m_to_idx(0.4 + 0.2 * (1-difficulty)), 2)    # narrow at high difficulty, must be >=0.4m
    stone_length = m_to_idx(0.6 + 0.3 * (1-difficulty))          # stones grow slightly as difficulty drops
    spacing = m_to_idx(1.7 - difficulty)        # center-to-center distance between stones; closer for low diff
    # To make them a little more unpredictable, randomly jitter some y positions
    y_jitter_ampl = max(m_to_idx(0.6 * difficulty), 1)           # up to 0.6m lateral displacement for hard

    # Set up initial flat spawn zone
    height_field[:alley_start, :] = 0
    goals[0] = [m_to_idx(1.0), mid_y]  # First goal is straight in front

    # Set the pit in the alley region
    height_field[alley_start:alley_end, :] = pit_depth

    # Lay stepping stones
    for i in range(num_stones):
        # Stone center position along x
        x = alley_start + i * spacing
        # Jitter y a bit except for first and last stones (to help ease-in/out)
        if i == 0 or i == num_stones - 1:
            y = mid_y
        else:
            y = mid_y + random.randint(-y_jitter_ampl, y_jitter_ampl)
        # Place the stone
        xs = slice(max(0, int(x - stone_length//2)), min(m_to_idx(length), int(x + (stone_length+1)//2)))
        ys = slice(max(0, int(y - stone_width//2)), min(m_to_idx(width), int(y + (stone_width+1)//2)))
        height_field[xs, ys] = 0  # Stones are at ground level

        # Place the goal in the center of each stone
        goals[i] = [int(x), int(y)]

    # Final exit ramp back to ground
    ramp_rng = range(alley_end, m_to_idx(length))
    for x in ramp_rng:
        h = pit_depth * (alley_end - x) / (alley_end - m_to_idx(length)-1)
        height_field[x, :] = h

    # Smoothly bring the ground to level at the end
    height_field[m_to_idx(length-1):, :] = 0

    return height_field, goals