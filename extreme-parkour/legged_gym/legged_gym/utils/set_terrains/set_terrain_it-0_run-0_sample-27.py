import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A zig-zag narrow beam course that tests lateral balance and precise turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for the "beam bridge"
    beam_height = 0.10 + 0.15 * difficulty  # 10-25cm tall beams
    pit_depth = -1.2                        # Pit under the beams to discourage falling off
    pit_start_x = m_to_idx(2.0)
    pit_end_x = m_to_idx(length - 2.0)
    beam_width = 0.42 - 0.17 * difficulty   # 0.42m (easy) to 0.25m (hard), challenging but possible (robot is 0.28m wide)
    beam_length = 2.5                       # Each beam straight section is 2.5m
    turn_length = 1.0                       # How far the path "moves over" per bend
    mid_y = m_to_idx(width / 2)

    # Pit area
    height_field[pit_start_x:pit_end_x, :] = pit_depth

    # Beam centerline directions (zig-zag: R, L, R...)
    beam_centers = []
    zig_y_positions = [
        mid_y, 
        mid_y + m_to_idx(turn_length), 
        mid_y - m_to_idx(turn_length), 
        mid_y + m_to_idx(turn_length), 
        mid_y - m_to_idx(turn_length)
    ]
    # Clamp to boundaries
    for i in range(len(zig_y_positions)):
        zig_y_positions[i] = np.clip(zig_y_positions[i], m_to_idx(beam_width/2)+1, m_to_idx(width)-m_to_idx(beam_width/2)-2)

    beam_x_starts = [pit_start_x + i*m_to_idx(beam_length) for i in range(5)]
    beam_x_ends = [x + m_to_idx(beam_length) for x in beam_x_starts]
    # Ensure last beam stops well before end
    for i in range(len(beam_x_ends)):
        beam_x_ends[i] = min(beam_x_ends[i], m_to_idx(length)-2)

    # Place the zig-zag beams ("narrow walkway by alternating right and left")
    for i in range(5):
        cx = (beam_x_starts[i] + beam_x_ends[i]) // 2
        cy = zig_y_positions[i]
        y1 = int(cy - m_to_idx(beam_width/2))
        y2 = int(cy + m_to_idx(beam_width/2))
        height_field[beam_x_starts[i]:beam_x_ends[i], y1:y2] = beam_height
        beam_centers.append( ( (beam_x_starts[i]+beam_x_ends[i])//2, (y1+y2)//2 ) )

    # Entry and exit ramps (allow for approach/egress on flat ground)
    spawn_x = m_to_idx(1)
    height_field[:pit_start_x, :] = 0.0
    height_field[pit_end_x:, :] = 0.0

    # -- Place 8 goals: entrance, after every turn, exit --
    # 0: Starting area (flat)
    goals[0] = [m_to_idx(0.8), mid_y]
    # 1: Start of first beam
    goals[1] = [beam_x_starts[0]+m_to_idx(0.1), zig_y_positions[0]]
    # 2: Middle of first beam
    goals[2] = [ (beam_x_starts[0]+beam_x_ends[0])//2, zig_y_positions[0]]
    # 3,4,5,6: After each turn
    for i in range(1,5):
        # At the start of each beam after a zig, place a goal
        goals[i+2] = [beam_x_starts[i]+m_to_idx(0.1), zig_y_positions[i]]
    # 7: Exit zone (end of last beam)
    goals[7] = [beam_x_ends[4] - m_to_idx(0.2), zig_y_positions[4]]

    return height_field, goals