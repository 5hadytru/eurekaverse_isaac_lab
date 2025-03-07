import random
import numpy as np

def set_terrain(length, width, field_resolution, difficulty):
    """Series of platforms with undulating connections for the robot to navigate and balance on."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Platform and undulation dimensions
    platform_length = np.random.uniform(0.8, 1.2 - 0.2 * difficulty)
    platform_length = m_to_idx(platform_length)
    platform_width = np.random.uniform(1.0, 1.5)
    platform_width = m_to_idx(platform_width)
    platform_height_min, platform_height_max = 0.1, 0.2 + 0.4 * difficulty
    undulation_amplitude = 0.05 + 0.1 * difficulty
    gap_length = np.random.uniform(0.2, 0.4 + 0.3 * difficulty)
    gap_length = m_to_idx(gap_length)
    
    mid_y = m_to_idx(width) // 2

    def add_platform(start_x, end_x, mid_y):
        half_width = platform_width // 2
        x1, x2 = start_x, end_x
        y1, y2 = mid_y - half_width, mid_y + half_width
        platform_height = np.random.uniform(platform_height_min, platform_height_max)
        height_field[x1:x2, y1:y2] = platform_height

    def add_undulation(start_x, end_x):
        x1, x2 = start_x, end_x
        y_indices = np.arange(0, m_to_idx(width))
        undulation_heights = undulation_amplitude * np.sin(np.linspace(0, 2 * np.pi, y_indices.size))
        height_field[x1:x2, :] += undulation_heights[np.newaxis, :]
    
    # Set spawn area to flat ground
    spawn_length = m_to_idx(2)
    height_field[0:spawn_length, :] = 0
    # Put first goal at spawn
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]
    
    cur_x = spawn_length

    for i in range(4):  # Set up alternating platforms and undulating terrain
        add_platform(cur_x, cur_x + platform_length, mid_y)
        goals[i+1] = [cur_x + platform_length / 2, mid_y]
        
        cur_x += platform_length
        cur_x += gap_length

        add_undulation(cur_x, cur_x + platform_length)
        goals[i+2] = [cur_x + platform_length / 2, mid_y]
        
        cur_x += platform_length
        cur_x += gap_length

    # Final goal placement
    goals[-1] = [cur_x + m_to_idx(0.5), mid_y]
    height_field[cur_x:, :] = 0

    return height_field, goals