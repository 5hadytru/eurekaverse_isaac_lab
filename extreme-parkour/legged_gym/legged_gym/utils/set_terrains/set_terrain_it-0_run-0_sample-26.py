import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of raised balance beams of varying widths to test precise foot placement and balancing."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # The robot spawns at (1, width/2), so all obstacles must be after x = 2
    spawn_length = m_to_idx(2)
    mid_y = m_to_idx(width // 2)

    # Beam heights increase with difficulty
    beam_height_min = 0.04 + 0.15 * difficulty  # At least robot's foot clearance
    beam_height_max = 0.09 + 0.28 * difficulty  # Challenging step-up at higher difficulty

    # Beam length and spacing settings
    # At hard, beams are longer, spacing (jump space) increases
    beam_length = 1.0 + 0.5 * difficulty        # meters
    beam_length = m_to_idx(beam_length)
    space_between = 0.5 + difficulty            # meters
    space_between = m_to_idx(space_between)

    # Beam width constricts with difficulty: at hardest, just wider than robot's foot span
    beam_width_max = 0.7 - 0.2 * difficulty     # meters
    beam_width_min = 0.35 + 0.1 * difficulty    # never narrower than 0.35m
    num_beams = 6

    # Set spawn area to flat ground
    height_field[0:spawn_length, :] = 0
    goals[0] = [spawn_length // 2, mid_y]

    # Function to add a balance beam at given x and y
    def add_beam(start_x, y_center, beam_len, beam_wid, height):
        x1 = int(start_x)
        x2 = min(int(start_x + beam_len), height_field.shape[0])
        y1 = max(0, int(y_center - beam_wid // 2))
        y2 = min(height_field.shape[1], int(y_center + (beam_wid + 1) // 2))
        height_field[x1:x2, y1:y2] = height  # Set beam height

    cur_x = spawn_length
    beam_centers_y = []
    turn_alternate = [0, 1] * 4
    for i in range(num_beams):
        # Place beam along varying y to encourage controlled narrow turns
        # Most beams are central, but some offset left/right
        if i == 0:
            y_shift = 0
        elif i == 1:
            # Encourage a gentle S-curve by lateral offsetting every other beam
            y_shift = int(m_to_idx(0.6 * (1 if random.random() > 0.5 else -1) * difficulty))
        else:
            last_y = beam_centers_y[-1]
            direction = (-1 if turn_alternate[i] else 1)
            y_shift = int(direction * m_to_idx(0.5 * difficulty))
            y_shift = np.clip(y_shift, -m_to_idx(1.0), m_to_idx(1.0))
        center_y = mid_y if i == 0 else np.clip(beam_centers_y[-1] + y_shift, m_to_idx(0.4), m_to_idx(width-0.4))

        beam_wid = m_to_idx(random.uniform(beam_width_min, beam_width_max))
        beam_ht = random.uniform(beam_height_min, beam_height_max)
        add_beam(cur_x, center_y, beam_length, beam_wid, beam_ht)

        # Place goal at the center of each beam
        goals[i + 1] = [cur_x + beam_length // 2, center_y]
        beam_centers_y.append(center_y)

        # Advance to the next beam, leave gap (pit)
        next_x = cur_x + beam_length + space_between

        # The gap between beams is deep pit to force foot placement and controlled jumps
        # Only create pit area if not last
        if i < num_beams - 1:
            pit_x1 = cur_x + beam_length
            pit_x2 = next_x
            height_field[int(pit_x1):int(pit_x2), :] = -1.2  # Pit is quite deep

        cur_x = next_x

    # Put the final goal on flat ground after the last beam
    final_goal_x = min(cur_x + m_to_idx(0.7), height_field.shape[0] - 1)
    height_field[int(cur_x):, :] = 0  # Flat ground at end
    goals[7] = [final_goal_x, beam_centers_y[-1]]

    return height_field, goals