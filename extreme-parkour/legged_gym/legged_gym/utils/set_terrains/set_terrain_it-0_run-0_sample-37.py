import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of alternating low hurdles (bars) and short stairs, testing stepping and climbing precision."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Configuration parameters (all scale with difficulty)
    mid_y = m_to_idx(width / 2)
    spawn_length = m_to_idx(2)
    height_field[:spawn_length, :] = 0.0  # Flat spawn area

    # Hurdle (bar) parameters:
    hurdle_height_min = 0.07 + 0.03 * difficulty     # lowest bar step at lowest diff (7cm), 10cm at hardest
    hurdle_height_max = 0.13 + 0.10 * difficulty     # up to 23cm at hardest
    hurdle_width = m_to_idx(1.2)                     # at least 1m wide

    # Stair parameters:
    stair_steps_min = 2
    stair_steps_max = 4
    stair_depth_per_step = 0.30                      # meters (enough for one stride)
    stair_rise_per_step = 0.07 + 0.06 * difficulty   # step height, 7-13cm per step

    n_obstacles = 6

    # Inter-obstacle gaps
    gap_base = 0.60 + 0.20 * difficulty    # horizontal ground between obstacles (in meters, scales with diff)
    gap_base_idx = m_to_idx(gap_base)
    bar_len = m_to_idx(0.08)               # thickness of bar/hurdle

    # The sequence: hurdle -> stairs -> hurdle -> stairs ...
    cur_x = spawn_length
    for i in range(n_obstacles):
        center_y = mid_y + random.randint(-m_to_idx(0.15), m_to_idx(0.15))

        # Place hurdle/bar
        if i % 2 == 0:
            # Randomize height for variety
            bar_height = np.random.uniform(hurdle_height_min, hurdle_height_max)
            # Hurdle position
            bar_x1 = cur_x
            bar_x2 = bar_x1 + bar_len
            bar_y1 = center_y - hurdle_width // 2
            bar_y2 = center_y + hurdle_width // 2
            height_field[bar_x1:bar_x2, bar_y1:bar_y2] = bar_height
            # Place goal just after the hurdle
            goals[i] = [bar_x2 + m_to_idx(0.20), center_y]
            # Allow quadruped some space to pass before next obstacle
            post_bar = gap_base_idx
            cur_x = bar_x2 + post_bar

        # Place stairs
        else:
            # Decide step parameters
            n_steps = np.random.randint(stair_steps_min, stair_steps_max + 1)
            step_depth = m_to_idx(stair_depth_per_step)
            step_rise = stair_rise_per_step
            stair_width = m_to_idx(1.2)
            stair_x1 = cur_x
            stair_x2 = stair_x1 + n_steps * step_depth
            stair_y1 = center_y - stair_width // 2
            stair_y2 = center_y + stair_width // 2
            # Build the steps (ascending)
            for j in range(n_steps):
                sx1 = stair_x1 + j * step_depth
                sx2 = sx1 + step_depth
                sh = (j+1) * step_rise
                height_field[sx1:sx2, stair_y1:stair_y2] = sh
            # Place goal at top step
            goals[i] = [stair_x2 - step_depth//2, center_y]
            # Advance to next
            cur_x = stair_x2 + gap_base_idx

    # Place final goals on last flat area (to ensure 8 in total)
    # If not enough obstacles, put extra goals at the end.
    for j in range(n_obstacles, 8):
        # Place equally spaced final goals at the end of the course
        final_x = min(cur_x + m_to_idx(0.4 * (j - n_obstacles + 1)), m_to_idx(length) - 2)
        goals[j] = [final_x, mid_y]

    # Ensure terrain after last obstacle is flat
    height_field[cur_x:, :] = 0.0

    # Clip any out-of-bounds (in rare case)
    height_field = height_field[:m_to_idx(length), :m_to_idx(width)]

    return height_field, goals