import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Narrow bridges and elevated platforms to test balance and navigation at varying heights."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Platform and bridge sizes
    platform_length = 1.0 - 0.3 * difficulty
    platform_length = m_to_idx(platform_length)
    platform_width = np.random.uniform(1.0, 1.2)
    platform_width = m_to_idx(platform_width)
    bridge_length_min, bridge_length_max = 2.0 - 0.8 * difficulty, 2.5 - 0.8 * difficulty
    bridge_length_min = m_to_idx(bridge_length_min)
    bridge_length_max = m_to_idx(bridge_length_max)
    bridge_width_min, bridge_width_max = 0.5, 0.6
    bridge_width_min = m_to_idx(bridge_width_min)
    bridge_width_max = m_to_idx(bridge_width_max)
    gap_length = 0.1 + 0.6 * difficulty
    gap_length = m_to_idx(gap_length)

    platform_height_min, platform_height_max = 0.2, 0.5 + 0.3 * difficulty

    mid_y = m_to_idx(width) // 2

    def add_platform(center_x, center_y):
        half_length = platform_length // 2
        half_width = platform_width // 2
        x1, x2 = center_x - half_length, center_x + half_length
        y1, y2 = center_y - half_width, center_y + half_width
        platform_height = np.random.uniform(platform_height_min, platform_height_max)
        height_field[x1:x2, y1:y2] = platform_height
    
    def add_bridge(start_x, end_x, center_y):
        half_width = random.randint(bridge_width_min, bridge_width_max) // 2
        x1, x2 = start_x, end_x
        y1, y2 = center_y - half_width, center_y + half_width
        bridge_height = random.uniform(platform_height_min, platform_height_max)
        height_field[x1:x2, y1:y2] = bridge_height

    # Set spawn area to flat ground
    spawn_length = m_to_idx(2)
    height_field[0:spawn_length, :] = 0
    # Put first goal at spawn
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    cur_x = spawn_length

    for i in range(6):  # Create 6 platforms and bridges
        # Add platform
        add_platform(cur_x + platform_length // 2, mid_y)
        goals[i] = [cur_x + platform_length // 2, mid_y]

        # Move x-coordinate ahead to place bridge
        cur_x += platform_length

        # Add a gap
        cur_x += gap_length

        # Add bridge
        bridge_length = random.randint(bridge_length_min, bridge_length_max)
        add_bridge(cur_x, cur_x + bridge_length, mid_y)
        
        # Move x-coordinate ahead to place next platform
        cur_x += bridge_length

        # Add another gap
        cur_x += gap_length
    
    # Add final platform
    add_platform(cur_x + platform_length // 2, mid_y)
    goals[6] = [cur_x + platform_length // 2, mid_y]

    # Add final goal behind the last platform, fill in the remaining gap
    goals[-1] = [cur_x + platform_length + m_to_idx(0.5), mid_y]
    height_field[cur_x + platform_length:, :] = 0

    return height_field, goals