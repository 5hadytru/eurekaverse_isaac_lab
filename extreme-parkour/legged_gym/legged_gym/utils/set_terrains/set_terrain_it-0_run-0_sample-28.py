import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of parallel balance beams with varying gap widths for quadruped agility and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Balance beam parameters
    # Difficulty increases: beams narrower and gaps wider
    min_beam_width = 0.25
    max_beam_width = 0.45
    beam_width = max_beam_width - difficulty * (max_beam_width - min_beam_width)  # [0.45m .. 0.25m]
    beam_width = max(beam_width, 0.22)  # Don't get too narrow
    beam_width_idx = m_to_idx(beam_width)

    beam_length = 1.5 # 1.5m per beam
    beam_length_idx = m_to_idx(beam_length)

    # Gap width between beams
    min_gap_width = 0.22
    max_gap_width = 0.7
    gap_width = min_gap_width + difficulty * (max_gap_width - min_gap_width)  # [0.22..0.7]
    gap_width_idx = m_to_idx(gap_width)

    # Beam elevation -- increase a bit for moderate challenge (always above 0)
    min_beam_height = 0.07 + difficulty * 0.07  # [0.07, 0.14]
    max_beam_height = 0.13 + difficulty * 0.12  # [0.13, 0.25]

    spacing = beam_length_idx + gap_width_idx
    total_beams = 8

    # The beams are placed in a zig-zag but always run lengthwise roughly along x (forward), some with slight offsets
    mid_y_idx = m_to_idx(width / 2)
    start_x_idx = m_to_idx(2)  # No obstacles before 2m, let robot spawn with clearance

    # For first 7 beams, rest of course is obstacle-free at end
    for i in range(7):
        x1 = start_x_idx + i * spacing
        x2 = x1 + beam_length_idx
        # Each beam placed at a slightly shifted y (zig-zag: [-0.4,0.4] meters)
        offset_y = random.uniform(-0.4, 0.4)
        y_center = mid_y_idx + m_to_idx(offset_y)
        y1 = max(y_center - beam_width_idx//2, 0)
        y2 = min(y_center + (beam_width_idx+1)//2, m_to_idx(width))

        # Randomize beam height within bounds for added challenge
        beam_height = random.uniform(min_beam_height, max_beam_height)
        height_field[x1:x2, y1:y2] = beam_height
        
        # The area below the beams is a pit with negative height (no climbing back onto beam)
        if i == 0:
            pit_x1 = x1
            pit_y1, pit_y2 = 0, m_to_idx(width)
            height_field[pit_x1:x2, pit_y1:pit_y2] = min(height_field[pit_x1:x2, pit_y1:pit_y2], -0.8 - 0.2*difficulty)

        else:
            prev_x2 = start_x_idx + (i-1) * spacing + beam_length_idx
            pit_start = prev_x2
            pit_end = x2
            height_field[pit_start:pit_end, 0:m_to_idx(width)] = -0.8 - 0.2*difficulty

        # Set goal in the center of each beam, about 60% along its length
        gx = x1 + int(0.6*beam_length_idx)
        gy = y_center
        if i < 7:
            goals[i] = [gx, gy]

    # Fill the area after the last beam, make it flat and set final goal at the end
    last_beam_x2 = start_x_idx + 6*spacing + beam_length_idx
    height_field[last_beam_x2:, :] = 0

    goals[7] = [last_beam_x2 + m_to_idx(0.6), mid_y_idx]  # final goal at course end

    # Set first goal at spawn, middle of width, after 1 meter
    goals[0] = [m_to_idx(1), mid_y_idx]
    # (Overwritten by beam 1 above; we leave this as in case the zig affects precision.)

    # Remove pits from spawn area
    height_field[:start_x_idx, :] = 0

    return height_field, goals