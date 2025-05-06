import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Alternating narrow balance beams and wide low platforms, testing balance control and precise foot placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Constants and robot size reference (0.645x0.28 meters)
    course_len_idx = m_to_idx(length)
    course_wid_idx = m_to_idx(width)
    spawn_x = m_to_idx(1)
    mid_y = course_wid_idx // 2
    num_beams = 4  # Odd goals at start/end of beams
    num_platforms = 4  # Even goals at end of platforms
    beam_length = 1.4 + difficulty * 1.3  # Beams get longer with difficulty
    beam_length_idx = m_to_idx(beam_length)
    beam_width = 0.45 + 0.15 * (1-difficulty)  # 0.6m at easy, 0.45m at hard (still safe)
    beam_width_idx = m_to_idx(beam_width)

    beam_height = 0.08 + 0.07 * difficulty  # Beams slightly above ground, up to 0.15m
    platform_length = 1.0 + difficulty * 0.7
    platform_length_idx = m_to_idx(platform_length)
    platform_width = 1.4 + 0.6 * (1-difficulty)  # Platforms wider at easy, min 1.4m
    platform_width_idx = m_to_idx(platform_width)
    platform_height = 0.05 + 0.02 * (random.random() - 0.5)  # Platforms flush or slightly above ground

    gap_length = 0.13 + difficulty * 0.22  # 0.13~0.35m gaps between obstacles
    gap_length_idx = m_to_idx(gap_length)

    # Set spawn area: at least 2m, clear of obstacles
    spawn_clear = m_to_idx(2.0)
    height_field[:spawn_clear, :] = 0
    goals[0] = [spawn_x, mid_y]  # First goal is right past spawn

    # Start building course, alternating beams and platforms, placing goals at obstacle ends
    cur_x = spawn_clear
    y_offset_choices = [0, m_to_idx(0.35), -m_to_idx(0.35)]  # Sometimes beams/platforms slightly laterally offset

    for i in range(1, 8):
        if i % 2 == 1:
            # Beam
            x1 = cur_x
            x2 = cur_x + beam_length_idx
            # Slightly vary y for challenge, prevent going out of bounds
            y_center = mid_y + random.choice(y_offset_choices)
            y1 = max(0, y_center - beam_width_idx // 2)
            y2 = min(course_wid_idx, y_center + int(np.ceil(beam_width_idx / 2)))
            # The rest of the ground at this x is a pit
            height_field[x1:x2, :] = -0.35 - 0.2 * difficulty
            # Draw the beam above the pit
            height_field[x1:x2, y1:y2] = beam_height
            # Set goal at the end of the beam
            goals[i] = [x2 - m_to_idx(0.18), (y1 + y2)//2]
            cur_x = x2 + gap_length_idx
        else:
            # Wide, safe platform
            x1 = cur_x
            x2 = cur_x + platform_length_idx
            y_center = mid_y + random.choice(y_offset_choices)
            y1 = max(0, y_center - platform_width_idx // 2)
            y2 = min(course_wid_idx, y_center + int(np.ceil(platform_width_idx / 2)))
            height_field[x1:x2, :] = 0  # Reset pit
            height_field[x1:x2, y1:y2] = platform_height
            # Set goal at the center of the platform
            goals[i] = [x2 - m_to_idx(0.20), (y1 + y2)//2]
            cur_x = x2 + gap_length_idx

    # Clear the final section
    height_field[cur_x:, :] = 0
    # Ensure last goal is in reach
    goals[7] = [min(cur_x + m_to_idx(0.3), course_len_idx - m_to_idx(0.3)), mid_y]

    # Guarantee all goals are within bounds and not inside pits
    for i in range(8):
        goals[i, 0] = np.clip(goals[i, 0], 0, course_len_idx-1)
        goals[i, 1] = np.clip(goals[i, 1], 0, course_wid_idx-1)

    return height_field, goals