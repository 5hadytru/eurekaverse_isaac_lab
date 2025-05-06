import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of narrow balance beams ('log bridges') and turning points, testing the robot's balance and precise foot placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for log beams
    # Beams become narrower and higher as difficulty increases
    log_length = 2.8 - 1.0 * difficulty  # meters
    log_width = 0.45 - 0.15 * difficulty  # meters (always >= 0.3)
    log_width = max(log_width, 0.3)
    log_height = 0.07 + 0.18 * difficulty  # meters
    beam_gap = 0.25 + 0.8 * difficulty  # meters, pit between beams

    spawn_area = m_to_idx(2)
    mid_y = m_to_idx(width / 2)
    course_length_idx = m_to_idx(length)
    course_width_idx = m_to_idx(width)
    
    # All of main course except spawn is set to pit depth
    pit_depth = -0.85 - 0.2 * difficulty
    height_field[spawn_area:, :] = pit_depth

    # Safe zone at the end
    safe_zone = m_to_idx(1.0)
    height_field[course_length_idx-safe_zone:, :] = 0

    # Place logs and goals
    beam_count = 4
    beams_start_x = spawn_area
    max_log_area = (course_length_idx - spawn_area - safe_zone) // beam_count - m_to_idx(0.08)
    log_length_idx = min(m_to_idx(log_length), max_log_area)
    log_width_idx = m_to_idx(log_width)
    gap_idx = m_to_idx(beam_gap)
    offset_angles = [0, np.pi/2, -np.pi/2, 0]  # To insert some 90deg turns
    beam_centers = []
    cur_x, cur_y = beams_start_x, mid_y

    for i in range(beam_count):
        theta = offset_angles[i]

        # For non-first beams, turn left/right if theta != 0
        if theta != 0:  # perform a turn
            if theta > 0:
                # turn left
                cur_y -= m_to_idx(1.1)
            else:
                # turn right
                cur_y += m_to_idx(1.1)
            cur_x += m_to_idx(0.35 + 0.22 * difficulty) # allow space before next beam

        # Compute ranges
        half_w = log_width_idx // 2
        half_l = log_length_idx // 2

        if theta == 0:
            # Beam is horizontal (x major)
            x1 = np.clip(cur_x, 0, course_length_idx-1)
            x2 = np.clip(cur_x + log_length_idx, 0, course_length_idx-1)
            y1 = np.clip(cur_y - half_w, 0, course_width_idx-1)
            y2 = np.clip(cur_y + half_w, 0, course_width_idx-1)
            height_field[x1:x2, y1:y2] = log_height
            beam_center_x = (x1 + x2) // 2
            beam_center_y = (y1 + y2) // 2
        else:
            # Beam is vertical (y major)
            x1 = np.clip(cur_x - half_w, 0, course_length_idx-1)
            x2 = np.clip(cur_x + half_w, 0, course_length_idx-1)
            y1 = np.clip(cur_y, 0, course_width_idx-1)
            y2 = np.clip(cur_y + log_length_idx, 0, course_width_idx-1)
            height_field[x1:x2, y1:y2] = log_height
            beam_center_x = (x1 + x2) // 2
            beam_center_y = (y1 + y2) // 2

        beam_centers.append((beam_center_x, beam_center_y))

        # Move x/y forward for next beam (after gap)
        if theta == 0:
            cur_x += log_length_idx + gap_idx
        else:
            cur_y += log_length_idx + gap_idx

    # Place the rest of the goals
    # First goal: just past spawn area, to get the bot centered on the first beam
    goals[0] = [m_to_idx(2.6), mid_y]
    # Next, beam transitions and turning points
    goals[1] = [beam_centers[0][0], beam_centers[0][1]]
    goals[2] = [beam_centers[0][0] + (beam_centers[1][0] - beam_centers[0][0]) // 2,
                beam_centers[0][1] + (beam_centers[1][1] - beam_centers[0][1]) // 2]
    goals[3] = [beam_centers[1][0], beam_centers[1][1]]
    goals[4] = [beam_centers[2][0], beam_centers[2][1]]
    goals[5] = [beam_centers[3][0], beam_centers[3][1]]
    # Penultimate goal: after last beam, before final safety zone
    goals[6] = [min(beam_centers[3][0] + m_to_idx(1.1), course_length_idx-2), beam_centers[3][1]]
    # Last goal: at the safe flat zone at the end
    goals[7] = [course_length_idx - m_to_idx(0.5), beam_centers[3][1]]

    return height_field, goals