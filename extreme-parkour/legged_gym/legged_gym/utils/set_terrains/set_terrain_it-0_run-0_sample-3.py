import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping-stone sequence of narrow balance beams testing the robot's balance and precision walking."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course parameters
    num_beams = 6
    # Balance beam properties scale with difficulty (hard: narrower & taller beams, wider gaps)
    beam_length = 2.2 - 0.4 * difficulty      # meters, 2.2 down to 1.8
    beam_width = 0.42 - 0.14 * difficulty     # meters, 0.42 down to 0.28 (just wider than robot width)
    beam_height = 0.08 + 0.22 * difficulty    # meters, 0.08 up to 0.30
    gap_length = 0.25 + 0.55 * difficulty     # meters, 0.25 up to 0.8
    lateral_offset_max = 0.9 - 0.6 * difficulty   # meters, at low difficulty the beams zigzag more

    # For safety, never let beams get narrower than robot width/2
    min_beam_width = 0.14
    beam_width = max(min_beam_width, beam_width)

    # Middle y-position of course
    mid_y = m_to_idx(width / 2)
    beam_length_idx = m_to_idx(beam_length)
    beam_width_idx = m_to_idx(beam_width)
    gap_length_idx = m_to_idx(gap_length)
    spawn_length = m_to_idx(2)

    # Keep area before first obstacle flat for the robot to spawn
    height_field[:spawn_length, :] = 0
    goals[0] = [m_to_idx(1), mid_y]  # spawn goal

    # All beams elevated above "pit" (pit is -1m for visual distinctiveness)
    height_field[spawn_length:, :] = -1.0

    cur_x = spawn_length
    for n in range(num_beams):
        # Zigzag: add a small random lateral (y) offset, less at high difficulty
        lateral = random.uniform(-lateral_offset_max, lateral_offset_max)
        beam_center_y = mid_y + m_to_idx(lateral)
        # Limit beam to course (min/max y)
        beam_y1 = np.clip(beam_center_y - beam_width_idx//2, 0, m_to_idx(width)-beam_width_idx)
        beam_y2 = beam_y1 + beam_width_idx
        # Place beam in x (ensure it fits in terrain)
        beam_x1 = cur_x
        beam_x2 = cur_x + beam_length_idx
        # Place beam as raised block
        height_field[beam_x1:beam_x2, beam_y1:beam_y2] = beam_height
        # Place the goal at beam center
        goals[n+1] = [beam_x1 + beam_length_idx // 2, beam_center_y]
        # Move to next beam location (add gap)
        cur_x = beam_x2 + gap_length_idx
        # If we're going beyond the terrain length, stop
        if cur_x + beam_length_idx >= m_to_idx(length):
            break

    # Final platform to step down at the end
    # Place last goal a bit before the end
    final_goal_x = min(cur_x + m_to_idx(0.5), m_to_idx(length)-1)
    goals[-1] = [final_goal_x, mid_y]
    # Safe exit platform at the end (flat ground at height 0)
    height_field[cur_x:, :] = 0

    return height_field, goals