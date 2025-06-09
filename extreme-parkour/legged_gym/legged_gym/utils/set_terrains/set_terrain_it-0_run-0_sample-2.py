import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of staggered balance beams across a shallow pit, testing lateral precision and balancing."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # Parameters for course layout
    course_length_idx = m_to_idx(length)
    course_width_idx = m_to_idx(width)
    spawn_x = m_to_idx(1)
    spawn_width = course_width_idx
    mid_y = course_width_idx // 2

    # Set spawn area to flat ground at height 0
    height_field[0:spawn_x, :] = 0

    # Beam parameters
    # Balance beams will get narrower and higher with difficulty
    num_beams = 4
    beam_length = 1.4    # meters (long enough for one stride and some balance)
    min_beam_width = 0.4 + 0.3 * (1-difficulty)   # can go down to 0.4m, up to 0.7m (easy)
    beam_height = 0.10 + 0.15 * difficulty  # from 0.1m up to 0.25m
    pit_depth = -0.18 - 0.2 * difficulty    # negative, from -0.18m to -0.38m
    gap_between_beams = 0.7 + 0.5 * difficulty  # increase gap with difficulty

    beam_length_idx = m_to_idx(beam_length)
    beam_height = float(beam_height)
    pit_depth = float(pit_depth)

    # Fill pit except at spawn and end section
    pit_start = spawn_x
    pit_end = m_to_idx(length-1)
    height_field[pit_start:pit_end, :] = pit_depth

    # Lay beams in a staggered formation (left/right)
    beam_centers_x = []
    beam_centers_y = []
    cur_x = spawn_x + m_to_idx(0.5)  # place first beam after spawn area
    lateral_offset = m_to_idx(0.8)  # how far beams can be from center
    for i in range(num_beams):
        frac = (i % 2)*2 - 1  # -1, 1, -1, ...
        center_y = mid_y + int(frac * (lateral_offset * (1-difficulty*0.7)))  # easier = larger offsets, harder = less
        beam_centers_x.append(cur_x + beam_length_idx // 2)
        beam_centers_y.append(center_y)

        # Determine width for this beam (gets narrower with each beam/difficulty)
        beam_width = min_beam_width - i * (min_beam_width-0.4)/max(1,num_beams-1)
        beam_width_idx = max(m_to_idx(beam_width), 4)  # at least 0.4m

        x1 = cur_x
        x2 = cur_x + beam_length_idx
        y1 = center_y - beam_width_idx//2
        y2 = center_y + (beam_width_idx+1)//2
        height_field[x1:x2, y1:y2] = beam_height  # place the beam

        # Place a goal at the middle of each beam
        goals[i+1] = [ (x1 + x2)//2, (y1 + y2)//2 ]

        # Step to next x
        cur_x = x2 + m_to_idx(gap_between_beams)

    # Place initial goal at spawn
    goals[0] = [m_to_idx(1)-m_to_idx(0.4), mid_y]

    # Final area: return to ground height at end
    height_field[cur_x:, :] = 0
    # Place the last goal at the end of the course, centered
    goals[-1] = [ min(cur_x + m_to_idx(0.7), course_length_idx-1), mid_y ]

    return height_field, goals