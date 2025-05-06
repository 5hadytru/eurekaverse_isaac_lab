import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Parallel balance beams of increasing narrowness for testing precise foot placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Balance beam parameters
    num_beams = 4
    # Beams heights: moderate height, enough to discourage falling
    beam_height = 0.2 + 0.25 * difficulty      # 0.2–0.45m
    beam_length = 2.0                          # Each beam is 2m long
    # Beams become narrower with difficulty
    min_beam_width = 0.18                      # min width never less than 18cm (smaller than feet span but traversable)
    max_beam_width = 0.45                      # starting width: at easy difficulty
    beam_width = max_beam_width - (max_beam_width-min_beam_width)*difficulty
    min_gap_w = 0.40                           # Minimum safe width for robot to turn at end

    # Beam configuration: zig-zag pattern
    x_gap = 0.3 + 0.3*difficulty               # spacing between beam ends
    y_margin = 0.2
    available_y = width - 2*y_margin
    beam_spacing = (available_y-(num_beams*beam_width))/(num_beams-1)
    beam_start_x = 2.1                         # leave 2.1m for spawn & turn-in
    beam_end_x = beam_start_x + beam_length

    # Set the starting area
    spawn_length = m_to_idx(2.0)
    height_field[0:spawn_length, :] = 0

    # Zig-zag beam path: robot must cross one beam, turn to next, etc
    y_positions = []
    for i in range(num_beams):
        y_positions.append(y_margin + i*(beam_width+beam_spacing) + beam_width/2.0)

    # Place the beams and the goals along the zig-zag path
    # Start
    goals[0] = [m_to_idx(1.0), m_to_idx(width/2.)]

    # Helper function
    def draw_beam(x0, x1, y, bw):
        y0 = m_to_idx(max(y-bw/2, 0))
        y1 = m_to_idx(min(y+bw/2, width))
        height_field[m_to_idx(x0):m_to_idx(x1), y0:y1] = beam_height

    # Place beams alternating left-right in y
    dir_sign = 1                           # direction along y axis; alternating zig-zag
    cur_x = beam_start_x
    for bi in range(num_beams):
        y_c = y_positions[bi]
        # Each beam
        draw_beam(cur_x, cur_x+beam_length, y_c, beam_width)

        # Place goal near end of this beam, before the turn
        xg = cur_x + beam_length - 0.3
        goals[bi+1] = [m_to_idx(xg), m_to_idx(y_c)]
        # Between beams: short flat segment + gap to next beam
        if bi < num_beams-1:
            # Flat pad for turning at end of beam
            pad_x0 = cur_x + beam_length
            pad_x1 = pad_x0 + x_gap
            y_next = y_positions[bi+1]
            # The turning pad connects both beams' y centers, wide enough for a turn
            y_min = min(y_c, y_next) - min_gap_w/2
            y_max = max(y_c, y_next) + min_gap_w/2
            height_field[m_to_idx(pad_x0):m_to_idx(pad_x1), m_to_idx(y_min):m_to_idx(y_max)] = beam_height

            # Place goal for the turn pad (alternate left/right on the pad)
            goals[bi+2] = [m_to_idx(pad_x0 + (x_gap/2)), m_to_idx(y_next)]
            cur_x = pad_x1
        # For the final beam, after loop add the last goal

    # After the final beam, make a broad goal area at the end
    final_x = cur_x + beam_length
    height_field[m_to_idx(cur_x):m_to_idx(final_x), :] = 0  # ground
    goals[7] = [m_to_idx(final_x - 0.5), m_to_idx(width/2.)]

    return height_field, goals