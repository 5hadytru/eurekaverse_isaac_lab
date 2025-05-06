import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Repeating narrow balance beams for quadrupedal balance and turning control."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- Course parameters ---
    # The beams are 0.4-0.5 meters wide (narrow, but allowed if no climbing)
    beam_width = 0.40 + 0.10 * (1 - difficulty)
    beam_width_idx = m_to_idx(beam_width)

    # Beams length: Cover the full width or length, but alternate orientation
    base_beam_length = 2.75 - 1.5 * difficulty    # Beams become shorter with difficulty
    gap_size = 0.25 + 0.55 * difficulty           # Gaps get larger with difficulty
    beam_height = 0.12 + 0.08 * difficulty        # Beams raised off ground more with difficulty

    length_idx = m_to_idx(length)
    width_idx  = m_to_idx(width)

    # Initial flat area (spawn)
    spawn_length = m_to_idx(2.0)
    height_field[:spawn_length, :] = 0.0
    mid_y = width_idx // 2

    # Set first goal in spawn zone, center
    goals[0] = [m_to_idx(1.0), mid_y]

    # Alternate between longitudinal and lateral beams, forcing turns 
    cur_x = spawn_length
    y_left = m_to_idx(0.7)
    y_right = width_idx - m_to_idx(0.7)
    x_reach = length_idx - m_to_idx(1.0)
    goal_ptr = 1
    orientation = 0  # 0: x-long/forward, 1: y-long/across

    for beam_num in range(1, 7): # 6 balance beams, 7 total transitions
        # Alternate sides
        if orientation == 0:
            # Forward beam
            beam_length = min(m_to_idx(base_beam_length + random.uniform(-0.25, 0.25)), x_reach - cur_x)
            x1, x2 = cur_x, cur_x + beam_length
            mid_beam_y = mid_y + random.randint(-m_to_idx(2.0) + beam_width_idx, m_to_idx(2.0) - beam_width_idx)
            y1 = mid_beam_y - beam_width_idx // 2
            y2 = mid_beam_y + beam_width_idx // 2
            # Draw the beam 
            height_field[x1:x2, y1:y2] = beam_height

            # Place next goal near end of this beam
            gx = int(x2 - beam_length // 3) if beam_num % 2 == 1 else int(x2 - 1)
            gy = int((y1 + y2) // 2)
            goals[goal_ptr] = [gx, gy]
            goal_ptr += 1

            # Add a lateral gap: all ground in front of this beam is dropped
            gap = m_to_idx(gap_size + random.uniform(-0.05, 0.06))
            cur_x = x2 + gap
            # Remove all height in the gap
            height_field[x2:cur_x, :] = -0.3 - 0.8 * difficulty
        else:
            # Lateral beam runs across y axis, small segment
            y_beam_center = random.randint(y_left + beam_width_idx // 2, y_right - beam_width_idx // 2)
            y1 = y_beam_center - beam_width_idx // 2
            y2 = y_beam_center + beam_width_idx // 2
            beam_length = m_to_idx(1.2 + 0.7 * (1 - difficulty))
            if cur_x + beam_length > x_reach:
                beam_length = x_reach - cur_x
            x1, x2 = cur_x, cur_x + beam_length
            height_field[x1:x2, y1:y2] = beam_height

            # Place next goal midway along the lateral beam, offset in y for a turn
            gx = int((x1 + x2) // 2)
            gy = int(y_beam_center)
            goals[goal_ptr] = [gx, gy]
            goal_ptr += 1

            # Add a forward gap after the beam
            gap = m_to_idx(gap_size + random.uniform(-0.05, 0.05))
            cur_x = x2 + gap
            height_field[x2:cur_x, :] = -0.3 - 0.8 * difficulty

        orientation = 1 - orientation  # Switch between orientations

        # Stop if close to course end
        if cur_x >= x_reach:
            break

    # --- Last goal: End zone ---
    end_x = min(length_idx - 1, cur_x + m_to_idx(0.7))
    height_field[end_x:, :] = 0.0
    goals[7] = [end_x - m_to_idx(0.4), mid_y]

    # If fewer than 8 goals have been placed, interpolate additional ones evenly
    final_goal_num = np.where(~(goals[:,0] == 0) & ~(goals[:,1] == 0))[0]
    if len(final_goal_num):
        last_goal = final_goal_num[-1] + 1
    else:
        last_goal = 1
    for g in range(last_goal, 7):
        # linearly place remaining goals along the central X trajectory
        gx = int(spawn_length + (end_x - spawn_length) * (g - last_goal + 1) / (7 - last_goal + 1))
        goals[g] = [gx, mid_y]

    return height_field, goals