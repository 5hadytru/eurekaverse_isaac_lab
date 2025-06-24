import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Sequence of balance beams and tight turns for testing dynamic balance and turning agility."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Terrain parameters
    course_length = m_to_idx(length)
    course_width = m_to_idx(width)
    mid_y = course_width // 2

    # Balance beam parameters
    beam_width = 0.25 + 0.15 * (1 - difficulty)       # wider beam at easy, narrower at hard, min 0.25m
    beam_length = 2.0 + 1.2 * difficulty              # longer beams at high difficulty
    beam_height = 0.08 + 0.12 * difficulty            # higher beams at high difficulty

    beam_width_idx = max(m_to_idx(beam_width), m_to_idx(0.25))
    beam_length_idx = m_to_idx(beam_length)
    beam_height = float(beam_height)

    # The robot spawns at x=1m, course starts after x=2m
    start_x_idx = m_to_idx(2.0)
    spawn_length = m_to_idx(2.0)
    # Flat ground around spawn
    height_field[:spawn_length, :] = 0

    # Transition step-up to the first beam
    step_up_length = m_to_idx(0.25)
    step_up_height = beam_height / 2
    height_field[start_x_idx:start_x_idx + step_up_length, :] = step_up_height

    # Beam arrangement: zigzag pattern (turn left, right, left...)
    zigzag_offset = 0.7           # meters left/right from center; how far the beam can go
    n_beams = 5
    gap_between_beams = 0.5 + 0.6 * difficulty        # gap between zigzags increases with difficulty
    gap_idx = m_to_idx(gap_between_beams)
    offset_idx = m_to_idx(zigzag_offset)
    zigzag_sign = 1

    beams = []
    x = start_x_idx + step_up_length
    for i in range(n_beams):
        # Calculate beam's y-center (zigzag)
        y_c = mid_y + zigzag_sign * offset_idx
        zigzag_sign *= -1  # alternate left/right
        # Bound within course
        y_c = np.clip(y_c, beam_width_idx // 2 + 1, course_width - (beam_width_idx // 2 + 1))
        x1 = int(x)
        x2 = min(int(x + beam_length_idx), course_length)
        y1 = int(y_c - beam_width_idx // 2)
        y2 = int(y_c + np.ceil(beam_width_idx / 2))
        # Set beam surface
        height_field[x1:x2, y1:y2] = beam_height
        beams.append((x1, x2, y_c))

        # Place corresponding goal at 60% along each beam
        gx = int(x1 + 0.6 * (x2 - x1))
        goals[i+1] = [gx, y_c]

        # Next beam starts after the gap, shifted along the x axis
        x = x2 + gap_idx

    # Set the starting goal at the start of the first beam (flat area)
    goals[0] = [m_to_idx(1.0), mid_y]

    # Place last beam as a final wide "platform" for rest/exit
    platform_length = m_to_idx(1.6)
    platform_height = beam_height * 1.1
    x1 = int(x)
    x2 = min(int(x + platform_length), course_length)
    y1 = int(mid_y - m_to_idx(0.8))
    y2 = int(mid_y + m_to_idx(0.8))
    height_field[x1:x2, y1:y2] = platform_height
    # Goal at center of platform
    goals[6] = [int((x1 + x2) // 2), mid_y]

    # Final goal near the end of the course, on flat ground as "exit"
    height_field[x2:, :] = 0
    goals[7] = [min(x2 + m_to_idx(0.5), course_length - 1), mid_y]

    # Fill unused goals (if any) as repeated on the platform
    for g in range(6, 8):
        if np.all(goals[g] == 0):
            goals[g] = goals[6]

    # Ensure all goals are in-bounds
    for i in range(goals.shape[0]):
        x, y = int(goals[i,0]), int(goals[i,1])
        goals[i,0] = np.clip(x, 0, course_length-1)
        goals[i,1] = np.clip(y, 0, course_width-1)

    return height_field, goals