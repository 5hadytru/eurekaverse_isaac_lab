import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping stone balance beams: a zig-zagging series of narrow elevated beams testing lateral precision and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))  # 8 sequential waypoints

    # Parameters for stepping beams
    beam_length = 1.8 - 0.6 * difficulty  # beams get shorter as difficulty increases
    beam_length = max(beam_length, 0.8)
    beam_length_idx = m_to_idx(beam_length)
    beam_width = 0.45 + 0.2 * (1-difficulty)  # narrower beams as difficulty increases, never below 0.45m
    beam_width = max(beam_width, 0.4)
    beam_width_idx = m_to_idx(beam_width)
    beam_height = 0.08 + 0.18 * difficulty   # up to 26cm elevation

    gap_length = 0.27 + 0.4 * difficulty     # gaps between beams
    gap_length_idx = m_to_idx(gap_length)

    # Zig-zag offset amplitude (how far beams deviate left/right)
    zigzag_amp = 0.5 + 0.9 * difficulty     # up to ~1.4m for hard tasks (still must stay within bounds)
    zigzag_amp_idx = m_to_idx(zigzag_amp)
    zigzag_sign = 1

    # Centerline index
    mid_y = m_to_idx(width/2)

    # Reserve spawn area as flat ground (from 0 to x=2m)
    spawn_length = m_to_idx(2)
    height_field[0:spawn_length, :] = 0

    # Place first goal at the start of the first beam, just ahead of spawn area
    cur_x = spawn_length
    cur_y = mid_y
    goals[0] = [spawn_length - m_to_idx(0.5), cur_y]

    for i in range(7):  # 7 beams, last goal off the beams
        # Zig-zag in the y direction
        zigzag = m_to_idx(zigzag_sign * random.uniform(0.3, zigzag_amp))
        # Ensure beam always stays within bounds
        beam_y_center = max(min(cur_y + zigzag, m_to_idx(width) - beam_width_idx//2 - 1), beam_width_idx//2)
        zigzag_sign *= -1  # Alternate sides

        # Define beam rectangle
        x1 = cur_x
        x2 = min(cur_x + beam_length_idx, m_to_idx(length)-1)
        y1 = int(beam_y_center - beam_width_idx//2)
        y2 = int(beam_y_center + beam_width_idx//2)
        # Place beam: raised above the ground
        height_field[x1:x2, y1:y2] = beam_height

        # Place goal at beam center (slightly forward to keep moving)
        goals[i+1] = [x1 + beam_length_idx//2, beam_y_center]

        # Next beam: add gap
        cur_x = x2 + gap_length_idx
        # Also move y-center for variation
        cur_y = beam_y_center

        # Optional: make the gap a "pit" for realism (if not final beam)
        if i < 6:  # Don't add pit after final beam
            pit_x1 = x2
            pit_x2 = min(x2 + gap_length_idx, m_to_idx(length)-1)
            height_field[pit_x1:pit_x2, :] = -0.4 - 0.3 * difficulty  # up to 70cm deep

    # Final goal is just off the last beam, where the robot can step down safely
    goals[7] = [min(cur_x + m_to_idx(0.5), m_to_idx(length)-2), mid_y]

    # Make sure the end area is flat so the robot finishes safely
    if cur_x < m_to_idx(length):
        height_field[cur_x:, :] = 0

    return height_field, goals