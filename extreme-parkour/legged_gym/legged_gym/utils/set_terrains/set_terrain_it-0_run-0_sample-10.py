import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """'Balancing Beams': A course of 6 narrow elevated beams offset laterally, requiring the quadruped to balance, walk, and change direction."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2)) 

    # Constants
    NUM_BEAMS = 6
    mid_y = m_to_idx(width / 2)
    spawn_length = m_to_idx(2.0)
    course_length = m_to_idx(length)
    course_width = m_to_idx(width)

    # Beam settings (challenging at difficulty=1, easy at difficulty=0)
    beam_length = 1.5 - 0.5 * difficulty             # 1-1.5 meters
    beam_length_idx = m_to_idx(beam_length)
    beam_width = 0.42 - 0.1 * difficulty             # 0.32-0.42 meters (just under the robot length, minimal turning room)
    beam_width_idx = max(m_to_idx(beam_width), m_to_idx(0.4))
    beam_height = 0.18 + 0.22 * difficulty           # 0.18-0.4 meters tall
    gap_length = 0.38 + 0.32 * difficulty            # 0.38-0.7 meters between beams

    y_offsets = np.linspace(-0.75, 0.75, NUM_BEAMS)  # Lateral offsetting of each beam for zig-zag path
    y_step = m_to_idx(0.5 + 0.5 * difficulty)        # Makes turns harder at higher difficulty

    # Clear the spawn area (flat ground)
    height_field[:spawn_length, :] = 0.0

    # Goals placement: initial
    goals[0] = [spawn_length // 2, mid_y]

    # Set up pit: robot must stay on the beams, don't allow walking on the floor
    height_field[spawn_length:, :] = -2.0

    cur_x = spawn_length
    for i in range(NUM_BEAMS):
        center_y = mid_y + int(y_offsets[i] * y_step)
        beam_x_start = cur_x
        beam_x_end   = beam_x_start + beam_length_idx
        beam_y_start = center_y - beam_width_idx // 2
        beam_y_end   = beam_y_start + beam_width_idx

        # Clamp bounds (for edge case at the ends)
        beam_x_start = max(0, beam_x_start)
        beam_x_end = min(height_field.shape[0], beam_x_end)
        beam_y_start = max(0, beam_y_start)
        beam_y_end = min(height_field.shape[1], beam_y_end)

        # Set beam HEIGHT
        height_field[beam_x_start:beam_x_end, beam_y_start:beam_y_end] = beam_height

        # Place goal at the center of each beam
        goal_x = beam_x_start + (beam_x_end - beam_x_start) // 2
        goal_y = center_y
        goals[i+1] = [goal_x, goal_y]

        # Progress to next beam start (add a gap)
        cur_x = beam_x_end + m_to_idx(gap_length)

    # Last transition: a goal on the ground at the end of the last beam
    final_x = min(course_length-1, cur_x + m_to_idx(0.7))
    goals[7] = [final_x, mid_y]

    # Fill beyond the last beam with flat ground for at least 1m
    height_field[cur_x:, :] = 0.0

    return height_field, goals