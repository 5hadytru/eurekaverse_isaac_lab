import random
import numpy as np

def set_terrain(length, width, field_resolution, difficulty):
    """Advanced mixed terrain with stepping stones, narrow beams, ramps, and variable height platforms to challenge navigation and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Set up platform dimensions
    platform_length = 1.0 - 0.4 * difficulty  # Platform length adjusting to difficulty
    platform_length = m_to_idx(platform_length)
    platform_width = np.random.uniform(0.4, 0.6)  # Narrower width to increase difficulty
    platform_width = m_to_idx(platform_width)
    platform_height_min, platform_height_max = 0.1 * difficulty, 0.4 * difficulty

    gap_length = 0.2 + 0.5 * difficulty  # Variable gaps
    gap_length = m_to_idx(gap_length)

    mid_y = m_to_idx(width) // 2

    def add_platform(start_x, end_x, mid_y):
        half_width = platform_width // 2
        x1, x2 = start_x, end_x
        y1, y2 = mid_y - half_width, mid_y + half_width
        platform_height = np.random.uniform(platform_height_min, platform_height_max)
        height_field[x1:x2, y1:y2] = platform_height

    dx_min, dx_max = -0.1, 0.2
    dx_min, dx_max = m_to_idx(dx_min), m_to_idx(dx_max)
    dy_min, dy_max = -0.3, 0.3
    dy_min, dy_max = m_to_idx(dy_min), m_to_idx(dy_max)

    # Set spawn area to flat ground
    spawn_length = m_to_idx(2)
    height_field[0:spawn_length, :] = 0

    # Put first goal at spawn
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]  

    # Baseline gap height for the rest of the pit
    height_field[spawn_length:, :] = -1.0

    cur_x = spawn_length
    for i in range(5):  # Set up 5 platforms or obstacles
        if i % 2 == 0:
            # Alternating between platforms and narrow beams
            dx = np.random.randint(dx_min, dx_max)
            dy = np.random.randint(dy_min, dy_max)
            add_platform(cur_x, cur_x + platform_length + dx, mid_y + dy)
            goals[i + 1] = [cur_x + (platform_length + dx) // 2, mid_y + dy]
        else:
            # Add narrow beam or ramp
            dx = np.random.randint(dx_min, dx_max)
            dy = np.random.randint(dy_min, dy_max)
            beam_width = np.random.uniform(0.2, 0.3)
            beam_width = m_to_idx(beam_width)
            half_width = beam_width // 2
            x1, x2 = cur_x, cur_x + platform_length + dx
            y1, y2 = (mid_y + dy) - half_width, (mid_y + dy) + half_width
            slope = np.linspace(0, np.random.uniform(platform_height_min, platform_height_max), x2 - x1)
            for i in range(x2 - x1):
                height_field[x1 + i, y1:y2] = slope[i]
            goals[i + 1] = [cur_x + (platform_length + dx) // 2, mid_y + dy]

        # Add gap
        cur_x += platform_length + dx + gap_length

    # Add final goal behind the last obstacle
    goals[-1] = [cur_x + m_to_idx(0.5), mid_y]
    height_field[cur_x:, :] = 0

    return height_field, goals