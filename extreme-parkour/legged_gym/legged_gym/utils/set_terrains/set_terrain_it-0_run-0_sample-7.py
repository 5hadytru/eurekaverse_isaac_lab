import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of staggered balance beams above a pit, requiring narrow-footed, precise traversal."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course parameters (balance beam skill test)
    # Balance beams are narrow, long planks, suspended above a pit
    beam_widths = np.linspace(0.45, 0.25 + 0.15 * (1-difficulty), 6)  # meters; more difficult = narrower
    beam_length = 1.6 + 0.3 * difficulty  # meters; slightly longer at high difficulty
    beam_height = 0.12 + 0.16 * difficulty  # meters
    pit_depth = -0.7 - 0.8 * difficulty     # meters; deeper pit at high difficulty
    spawn_length = m_to_idx(2)
    mid_y = m_to_idx(width) // 2

    # Helper to add a beam, returns center (x, y) for goals
    def add_beam(center_x, center_y, length, width, height):
        lx = m_to_idx(length / 2)
        wx = m_to_idx(width / 2)
        x1, x2 = max(0, center_x-lx), min(m_to_idx(length), center_x+lx)
        y1, y2 = max(0, center_y-wx), min(m_to_idx(width), center_y+wx)
        height_field[x1:x2, y1:y2] = height
        return (center_x, center_y)

    # Set pit, preserve flat ground at start and finish
    height_field[spawn_length:, :] = pit_depth
    finish_length = m_to_idx(0.8)
    height_field[-finish_length:, :] = 0

    # Set spawn goal
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    # Parameters for staggering the beams
    zigzag_displacement = 0.5 + 0.3 * difficulty
    zigzag_displacement = min(zigzag_displacement, (width / 2) - min(beam_widths)/2 - 0.15)
    beam_start = spawn_length + m_to_idx(0.5)  # leave some runway before first beam
    inter_beam_gap = 0.4 + 0.4 * difficulty

    # Place six beams, staggered left and right
    cur_x = beam_start
    y_center = mid_y
    for i in range(6):
        width_beam = beam_widths[i]
        # Zig-zag: alternate sign for lateral offset
        sign = -1 if i % 2 else 1
        offset = sign * m_to_idx(zigzag_displacement)
        y_center_beam = int(np.clip(mid_y + offset, m_to_idx(width_beam/2)+2, m_to_idx(width)-m_to_idx(width_beam/2)-2))

        # Place beam
        beam_cx = int(cur_x + m_to_idx(beam_length/2))
        add_beam(beam_cx, y_center_beam, beam_length, width_beam, beam_height)
        # Set goal at center of beam
        goals[i+1] = [beam_cx, y_center_beam]

        # Next beam further along x
        cur_x += m_to_idx(beam_length + inter_beam_gap)
        # Next beam zig-zags laterally

    # Final goal after last beam (return to flat ground)
    final_goal_x = min(cur_x + m_to_idx(0.7), m_to_idx(length) - m_to_idx(0.6))
    goals[7] = [final_goal_x, mid_y]
    height_field[final_goal_x:, :] = 0

    return height_field, goals