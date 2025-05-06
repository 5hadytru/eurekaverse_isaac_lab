import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone 'log balance beam' course: Robot must traverse a sequence of long, narrow, slightly wobbly beams, with gaps in between that increase in length and narrowness at higher difficulty."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for the beams (balance logs)
    # Beam settings scale with difficulty
    min_beam_width = 1.1 - 0.5 * difficulty  # [1.1, 0.6] m, always > quadruped width, but becomes more challenging
    max_beam_width = 1.5 - 0.5 * difficulty  # widest beams first
    min_beam_length = 1.8 - 0.7 * difficulty  # [1.8, 1.1] m
    max_beam_length = 2.6 - 1.0 * difficulty  # [2.6, 1.6] m
    beam_height = 0.08 + 0.20 * difficulty    # [0.08, 0.28] m

    # Gaps between beams become wider/harder
    min_gap = 0.15 + 0.25 * difficulty   # [0.15, 0.4] m
    max_gap = 0.35 + 0.45 * difficulty   # [0.35, 0.8] m

    field_x, field_y = m_to_idx(length), m_to_idx(width)
    mid_y = field_y // 2
    spawn_length = m_to_idx(2)

    # Start with safe flat area for spawn
    height_field[:spawn_length, :] = 0
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]  # first goal, just ahead of spawn

    # The region after spawn area is shallow "pit" (low or slightly negative, hard to walk, encourages beam usage)
    height_field[spawn_length:, :] = -0.15 - difficulty * 0.35

    # Helper to add a beam as a raised platform
    def add_beam(x0, x1, y_center, beam_w, h):
        y0 = int(max(0, y_center - beam_w // 2))
        y1 = int(min(field_y, y_center + beam_w // 2))
        height_field[x0:x1, y0:y1] = h

    # Course: sequence of 6 stepped beams each with a goal on/after
    cur_x = int(spawn_length)
    used_beams = []

    for i in range(1, 7):  # 6 beams/obstacles
        l = m_to_idx(np.random.uniform(min_beam_length, max_beam_length))
        w = m_to_idx(np.random.uniform(min_beam_width, max_beam_width))
        if w < m_to_idx(0.4):  # don't allow too narrow for balance beam
            w = m_to_idx(0.4)
        h = beam_height + np.random.uniform(-0.01, 0.03)  # add tiny random "wobble" in beam height
        y_shift = int(random.uniform(-field_y // 4 * 0.5 * difficulty, field_y // 4 * 0.5 * difficulty))  # mild zig-zag at higher difficulty
        beam_center_y = mid_y + y_shift

        # Place beam (make sure it's within bounds)
        x0, x1 = int(cur_x), int(min(cur_x + l, field_x - 2))
        add_beam(x0, x1, beam_center_y, w, h)
        used_beams.append((x0, x1, beam_center_y, w, h))

        # Place ith goal around 2/3rds of beam's x, center y
        bx_center = int(x0 + 0.66 * (x1 - x0))
        goals[i] = [bx_center, beam_center_y]

        # Gap to next beam
        gap = m_to_idx(np.random.uniform(min_gap, max_gap))
        cur_x = int(x1 + gap)
        if cur_x >= field_x - m_to_idx(1.0):
            break

    # Final goal: put it at course end straight after last beam, on flat ground
    final_goal_x = min(field_x - m_to_idx(1), cur_x + m_to_idx(0.6))
    goals[7] = [final_goal_x, mid_y]

    # Fill out any unused goals (for e.g. if the ends up < 8 beams)
    for idx in range(1, 7):
        if np.all(goals[idx] == 0):
            # Default to somewhere after spawn
            goals[idx] = [spawn_length + m_to_idx(2.0 * idx), mid_y]

    # Restore last area to be flat (for final goal)
    height_field[final_goal_x:, :] = 0

    return height_field, goals