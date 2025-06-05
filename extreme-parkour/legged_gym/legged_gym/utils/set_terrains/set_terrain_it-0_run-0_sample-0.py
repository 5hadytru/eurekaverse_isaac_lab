import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A balance beam course: Long, narrow beams (at ground level) and wide low platforms alternating, requiring the robot to walk steadily, turn, and step between beams and platforms."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # Course constants (robot size: 0.645 x 0.28 m)
    beam_width = (0.3 + 0.25 * (1-difficulty))        # 0.55m at easy, 0.3m at hard
    beam_width = max(beam_width, 0.3)
    beam_width_idx = m_to_idx(beam_width)
    beam_length = 2.2 + 1.5 * difficulty              # 2.2m at easy, up to 3.7m at hard
    beam_length_idx = m_to_idx(beam_length)
    platform_size = 1.4 - 0.6 * difficulty            # 1.4m wide at easy, down to 0.8m at hard
    platform_size_idx = m_to_idx(platform_size)
    beam_height = 0.05 + 0.08 * difficulty            # 5cm at easy, 13cm at hard
    platform_height = 0.00                            # flush with ground

    course_x = m_to_idx(1.8)  # Start after spawn (avoid [0,2m))
    alternating_offset = m_to_idx(1.3)                # Lateral offset for turns

    cur_y_idx = m_to_idx(width / 2)                   # Center line
    width_idx = m_to_idx(width)

    # Helper for adding beam ("balance beam" obstacle)
    def add_beam(x_start, x_end, y_center, width_idx, height):
        y1 = max(y_center - width_idx // 2, 0)
        y2 = min(y_center + width_idx // 2 + 1, height_field.shape[1])
        x1 = int(x_start)
        x2 = int(min(x_end, height_field.shape[0]))
        height_field[x1:x2, y1:y2] = height

    # Helper for adding platform (wider square/circular area)
    def add_platform(x_center, y_center, size_idx, height):
        half = size_idx // 2
        x1 = int(max(x_center - half, 0))
        x2 = int(min(x_center + half + 1, height_field.shape[0]))
        y1 = int(max(y_center - half, 0))
        y2 = int(min(y_center + half + 1, height_field.shape[1]))
        height_field[x1:x2, y1:y2] = height

    # Set spawn area to flat ground
    spawn_length = m_to_idx(2.0)
    height_field[:spawn_length, :] = 0

    # GOAL 1: On first platform, after spawn
    plat_x_center = course_x
    plat_y_center = cur_y_idx
    add_platform(plat_x_center, plat_y_center, platform_size_idx, platform_height)
    goals[0, :] = [plat_x_center, plat_y_center]

    # GOAL 2: Cross first balance beam, straight
    beam_x_start = plat_x_center + platform_size_idx // 2
    beam_x_end = beam_x_start + beam_length_idx
    beam_y_center = plat_y_center
    add_beam(beam_x_start, beam_x_end, beam_y_center, beam_width_idx, beam_height)
    goals[1, :] = [(beam_x_start + beam_x_end) // 2, beam_y_center]
    
    # GOAL 3: On second platform (now offset to one side, requires turn)
    plat2_x_center = beam_x_end + platform_size_idx // 2
    is_left = True
    offset_y = (-alternating_offset if is_left else alternating_offset)
    plat2_y_center = plat_y_center + offset_y
    add_platform(plat2_x_center, plat2_y_center, platform_size_idx, platform_height)
    goals[2, :] = [plat2_x_center, plat2_y_center]

    # GOAL 4: Second, angled balance beam (turn + beam)
    # This beam runs at an angle requiring turning and correcting
    angle_sign = -1 if is_left else 1
    beam2_x_start = plat2_x_center + platform_size_idx // 2
    beam2_x_end = beam2_x_start + beam_length_idx
    # y ~ y0 + m(x-x0)
    x_indices = np.arange(beam2_x_start, min(beam2_x_end, height_field.shape[0]))
    diagonal_slope = (alternating_offset / beam_length_idx) * angle_sign
    beam2_y_centerline = plat2_y_center + diagonal_slope * (x_indices - beam2_x_start)
    for idx, xi in enumerate(x_indices):
        y_c = int(np.round(beam2_y_centerline[idx]))
        y1 = max(y_c - beam_width_idx // 2, 0)
        y2 = min(y_c + beam_width_idx // 2 + 1, height_field.shape[1])
        height_field[xi, y1:y2] = beam_height
    goals[3, :] = [(beam2_x_start + beam2_x_end) // 2, int(plat2_y_center + diagonal_slope * ((beam2_x_end-beam2_x_start)//2))]

    # GOAL 5: Final wide platform (centered again)
    plat3_x_center = beam2_x_end + platform_size_idx // 2
    plat3_y_center = m_to_idx(width / 2)
    add_platform(plat3_x_center, plat3_y_center, platform_size_idx, platform_height)
    goals[4, :] = [plat3_x_center, plat3_y_center]

    # Fallback: ensure all indices are inside bounds and integers
    goals = np.clip(np.round(goals).astype(np.int16), [0,0], [height_field.shape[0]-1, height_field.shape[1]-1])

    return height_field, goals