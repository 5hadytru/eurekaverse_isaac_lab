import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of sloped ramps (A-frame dogwalks) to test balance and walking on angled surfaces."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Terrain parameters
    mid_y = m_to_idx(width // 2 if width % 2 == 0 else width / 2)
    course_x = 0

    # Ramp parameters
    ramp_base_length = 2.0           # meters
    ramp_height = 0.25 + 0.25 * difficulty   # meters, ramps are 0.25-0.5m tall
    ramp_length = 1.25 + 0.75 * difficulty   # meters, each sloped section is 1.25-2.0m long
    flat_top_length = 0.35 + 0.2 * difficulty # meters, length of flat plate at ramp peak
    ramp_total_length = 2 * ramp_length + flat_top_length

    ramp_width = 1.2 - 0.35 * difficulty # 1.2m at easy, 0.85m at hard
    ramp_width = max(1.0, ramp_width)    # never less than 1m
    side_clear = m_to_idx(0.2)           # always at least 0.2m to terrain edge

    num_ramps = 5
    total_used_x = num_ramps * ramp_total_length + (num_ramps - 1) * 0.5
    # If more space, use gentler slopes. If less, stack them closer
    inter_ramp_gap = (length - num_ramps * ramp_total_length) / (num_ramps + 1)
    inter_ramp_gap = max(0.3, inter_ramp_gap)  # no less than 0.3m between ramps

    cur_x = m_to_idx(2)  # Start after spawn area
    height_field[:cur_x, :] = 0.0  # Flat spawn area
    goals[0] = [m_to_idx(1), mid_y]  # First goal at center of spawn

    # Allows some variance in y offset, but keep most ramps nearly straight
    y_offsets = np.linspace(0, m_to_idx(width - ramp_width) // 2, num_ramps, dtype=int)

    def add_dogwalk(x_start, y_center, ramp_len, flat_len, width, height):
        """Add an A-frame ramp with a flat top centered at y_center."""
        y1 = y_center - m_to_idx(width // 2)
        y2 = y_center + m_to_idx(width // 2)

        # Ramp up
        for step in range(m_to_idx(ramp_len)):
            x = x_start + step
            h = step / m_to_idx(ramp_len) * height
            height_field[x, y1:y2] = h

        # Flat top
        for step in range(m_to_idx(flat_len)):
            x = x_start + m_to_idx(ramp_len) + step
            height_field[x, y1:y2] = height

        # Ramp down
        for step in range(m_to_idx(ramp_len)):
            x = x_start + m_to_idx(ramp_len) + m_to_idx(flat_len) + step
            h = height * (1 - step / m_to_idx(ramp_len))
            height_field[x, y1:y2] = h

        return ((x_start + m_to_idx(ramp_len + flat_len // 2)), y_center)

    for i in range(num_ramps):
        y_center = mid_y + random.randint(-side_clear, side_clear)
        ramp_center_x = cur_x + m_to_idx(ramp_length + flat_top_length / 2)
        goal_x = cur_x + m_to_idx(ramp_length + flat_top_length / 2)
        goal_y = y_center
        goals[i+1] = [goal_x, goal_y]
        # Each ramp is an A-frame "dogwalk"
        add_dogwalk(cur_x, y_center, ramp_length, flat_top_length, ramp_width, ramp_height)
        cur_x += m_to_idx(ramp_total_length + inter_ramp_gap)

    # After last ramp, flat ground, final goal
    height_field[cur_x:, :] = 0.0
    goals[6] = [cur_x + m_to_idx(0.1), mid_y]
    # 7th goal at very end of terrain
    goals[7] = [m_to_idx(length) - m_to_idx(1), mid_y]

    return height_field, goals