import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of balanced beams (narrow walkways) elevated above deep pits, for precise foot placement and balancing."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    ## Parameters for beams
    beam_num = 6
    # Beam width: challenging precision, at least 0.42 m (about 1.5x robot width), never <0.4m
    beam_width = 0.42 - 0.1 * difficulty  # Narrower as difficulty increases, but never <0.4
    beam_width = max(beam_width, 0.4)
    beam_width_idx = m_to_idx(beam_width)
    # Beam height: 0.12-0.35 (tall enough to punish falling), up at higher difficulty
    beam_height = 0.12 + 0.23 * difficulty
    # Space between beams (pit): 0.50 (easy) to 1.60 (hard) meters
    pit_width = 0.5 + 1.10 * difficulty
    pit_width_idx = m_to_idx(pit_width)
    # Beam length: fill area between pits
    usable_length = length - 2   # leave 2m for spawn and exit
    beam_length = usable_length / beam_num
    beam_length_idx = m_to_idx(beam_length)

    # Beams may be slightly staggered left or right
    mid_y = m_to_idx(width / 2)
    max_offset = m_to_idx((width - beam_width) / 2 - 0.1)   # beams must always fit

    # Set pit depth: deep enough to force failed attempt without climbing out
    pit_depth = -0.6 - 0.6 * difficulty

    # Set spawn and end flat
    spawn_len_idx = m_to_idx(2)
    end_len_idx = m_to_idx(1)
    height_field[:spawn_len_idx, :] = 0
    height_field[-end_len_idx:, :] = 0

    # Set goals
    start_x = spawn_len_idx
    cur_x = start_x
    step_idx_list = []

    for i in range(beam_num):
        # Random offset
        offset = random.randint(-max_offset, max_offset)
        y0 = mid_y + offset - beam_width_idx // 2
        y1 = mid_y + offset + (beam_width_idx + 1) // 2
        x0 = cur_x
        x1 = min(cur_x + beam_length_idx, height_field.shape[0] - end_len_idx)
        # Draw the beam
        height_field[x0:x1, y0:y1] = beam_height
        # Set the goal at 2/3 along the beam, centered
        goal_x = x0 + int((x1-x0) * np.clip(0.66 + 0.15*(random.random()-0.5), 0.55, 0.75))
        goal_y = (y0 + y1) // 2
        goals[i+1] = [goal_x, goal_y]
        step_idx_list.append((x0, x1, y0, y1))
        # Next pit
        cur_x = x1
        # Draw the pit so that everywhere off-beam is a pit
        height_field[x1:x1+pit_width_idx, :] = pit_depth
        cur_x += pit_width_idx

    # Set final goal at the beginning of the last flat section (exit)
    goals[0] = [m_to_idx(1.0), mid_y] # spawn
    goals[beam_num+1] = [min(height_field.shape[0] - m_to_idx(0.5), height_field.shape[0] - 2), mid_y]

    # Fill leftovers in goals to meet the 8 total
    for i in range(beam_num+2, 8):
        # Just repeat exit goal
        goals[i] = goals[beam_num+1]

    # Ensure beam edges are not cut off
    height_field[:spawn_len_idx, :] = 0 # clear spawn again if any overlap

    return height_field, goals