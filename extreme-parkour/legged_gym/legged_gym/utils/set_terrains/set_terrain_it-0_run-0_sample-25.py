import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Alternating balance beams and narrow stepping stones for quadruped dynamic stability and precise paw placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters
    mid_y = m_to_idx(width) // 2
    spawn_length = m_to_idx(2.0)
    n_obstacles = 4  # 4 beams, 4 stone gaps = 8 segments before finish
    start_x = spawn_length
    finish_buffer = m_to_idx(1.0)

    # Base ground is a shallow pit, so off-beam is a penalty
    height_field[spawn_length:, :] = -0.15 - 0.25*difficulty  # deeper pit at higher difficulty

    # Function for adding a balance beam: narrow, long, elevated "bridge"
    def add_beam(x1, x2, y_center, width, height):
        y1 = int(np.clip(y_center - width // 2, 0, height_field.shape[1]))
        y2 = int(np.clip(y_center + width // 2, 0, height_field.shape[1]))
        height_field[x1:x2, y1:y2] = height

    # Function for adding stepping stone: short, small round platform
    def add_stepping_stone(x_center, y_center, r, height):
        x_idx = np.arange(height_field.shape[0])
        y_idx = np.arange(height_field.shape[1])
        X, Y = np.meshgrid(x_idx, y_idx, indexing='ij')
        mask = ((X - x_center) ** 2 + (Y - y_center) ** 2) <= r ** 2
        height_field[mask] = height

    # Beam and stone sizes scale with difficulty
    beam_length = m_to_idx(1.7 - 0.3*difficulty)
    beam_width = m_to_idx(1.0 - 0.55*difficulty)  # as thin as 45cm at max difficulty
    beam_height = 0.0 + 0.05 + 0.15*difficulty    # 5-20cm above base

    stone_radius = m_to_idx(0.3 - 0.08*difficulty)        # 30-22cm radius (so 44cm minimum width)
    stone_height = 0.0 + 0.08 + 0.18*difficulty           # 8-26cm height
    stone_gap = m_to_idx(0.4 + 0.3*difficulty)            # distance between stone centers

    cur_x = start_x
    section_spacing = m_to_idx(0.3)   # gap between last stone/beam and next beam/stone

    # Start with spawn area flat
    height_field[:spawn_length, :] = 0.0
    goals[0] = [m_to_idx(0.8), mid_y]     # first goal is just past spawn

    # Alternate beams and stones
    goal_idx = 1
    for seg in range(n_obstacles):
        # 1. Add balance beam
        next_x = min(cur_x + beam_length, height_field.shape[0]-finish_buffer)
        beam_y = mid_y + m_to_idx( random.uniform(-0.7 + 1.4*random.random(), 0.7 - 1.4*random.random()) )
        add_beam(cur_x, next_x, beam_y, beam_width, beam_height)

        # Place goal mid-beam
        goals[goal_idx] = [cur_x + (next_x-cur_x)//2, beam_y]
        goal_idx += 1

        cur_x = next_x + section_spacing

        # 2. Add stepping stones (3 stones across the pit)
        if seg < n_obstacles: # last segment does not need more stones after finish
            n_stones = 3
            stones_y = np.linspace(mid_y - m_to_idx(0.7), mid_y + m_to_idx(0.7), n_stones)
            stones_x = [cur_x + i*stone_gap for i in range(n_stones)]
            for x_c, y_c in zip(stones_x, stones_y):
                add_stepping_stone(int(x_c), int(y_c), stone_radius, stone_height)

            # Place goal at farthest stone
            goals[goal_idx] = [stones_x[-1], stones_y[-1]]
            goal_idx += 1

            # Prepare for next beam
            cur_x = int(stones_x[-1]) + section_spacing

    # Final safe zone
    height_field[cur_x:, :] = 0.0
    # Place last goal at finish area
    goals[-1] = [min(cur_x + m_to_idx(0.9), height_field.shape[0]-1), mid_y]

    return height_field, goals