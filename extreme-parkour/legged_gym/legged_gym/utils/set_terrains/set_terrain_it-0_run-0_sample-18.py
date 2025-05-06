import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of tilted balance beams to challenge the robot's dynamic walking and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Balance beam parameters
    # Each beam spans the course width; their heights and slopes increase with difficulty
    beam_length = 1.7 - 0.4 * difficulty   # shorter beams for higher difficulty
    beam_width = 0.42 + 0.25 * (1-difficulty) # slightly wider at low difficulty
    beam_start_height = 0.07 + 0.09 * difficulty
    beam_max_slope = 0.06 + 0.17 * difficulty     # meters elevation over length

    gap_len = 0.27 + 0.48 * difficulty
    num_beams = 6

    total_beam_and_gaps = num_beams * m_to_idx(beam_length) + (num_beams-1) * m_to_idx(gap_len)
    available_length = m_to_idx(length) - m_to_idx(2) - m_to_idx(1) # leave spawn and exit areas

    # Distribute beams and gaps in sequence along x-axis
    x = m_to_idx(2)   # Start after safe spawn zone
    mid_y = m_to_idx(width/2)

    def add_beam(start_x, beam_len, center_y, slope, height_start):
        """Adds a sloped balance beam."""
        beam_width_idx = m_to_idx(beam_width) // 2
        for i in range(m_to_idx(beam_len)):
            h = height_start + slope * (i)
            height_field[start_x+i, center_y-beam_width_idx:center_y+beam_width_idx] = h

    # Set spawn area flat
    height_field[:x, :] = 0
    goals[0] = [x - m_to_idx(1), mid_y] # first goal at the end of spawn area

    for i in range(num_beams):
        # Random slope direction for variety (+ up, - down)
        slope = random.choice([1, -1]) * (beam_max_slope / m_to_idx(beam_length))
        # Random left/right offset (beam center y), small at low difficulty
        beam_offset_range = (0.10 + 0.70 * difficulty) * (width/2 - beam_width/2 - 0.3)
        beam_center_y = mid_y + m_to_idx(random.uniform(-beam_offset_range, beam_offset_range))
        height_start = beam_start_height + random.uniform(-0.03, 0.03)*difficulty
        # Add narrow, sloped beam
        add_beam(x, beam_length, beam_center_y, slope, height_start)
        # Set next goal at middle of beam
        beam_center_x = x + m_to_idx(beam_length/2)
        goals[i+1] = [beam_center_x, beam_center_y]
        # Set gaps between beams to be shallow pit
        pit_x1 = x + m_to_idx(beam_length)
        pit_x2 = pit_x1 + m_to_idx(gap_len)
        if pit_x2 < m_to_idx(length) - m_to_idx(1):
            height_field[pit_x1:pit_x2, :] = -0.22 - 0.15*difficulty
        x = pit_x2

    # Final goal on ground at end of corridor
    if x < m_to_idx(length):
        height_field[x:, :] = 0
        goals[7] = [x + m_to_idx(0.6), mid_y]
    else:
        goals[7] = [m_to_idx(length)-1, mid_y]

    return height_field, goals