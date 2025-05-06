import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of low, narrow balance beams suspended over pits, testing lateral precision and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))
    
    # Main parameters
    spawn_length = m_to_idx(2.0)
    terrain_length = m_to_idx(length)
    terrain_width = m_to_idx(width)
    mid_y = terrain_width // 2

    # Balance beam specs
    # As difficulty increases, beams become narrower and gaps become wider
    beam_length = 1.6  # meters, fixed so robot has to stay balanced for a span
    min_beam_width = 0.6   # meters, at highest difficulty the beam is fairly narrow for this robot
    max_beam_width = 1.2
    beam_width = max(min_beam_width, max_beam_width - (max_beam_width - min_beam_width) * difficulty)
    beam_height = 0.08 + 0.05 * difficulty   # slightly higher beams with more diff

    min_gap = 0.2   # meters, easy = beams nearly touch
    max_gap = 0.8   # meters, hard = real jump required
    gap_length = min_gap + (max_gap - min_gap) * difficulty

    n_beams = 6
    # Make beams alternate sides so the robot must sidestep from beam to beam
    lateral_range = width - beam_width - 0.28  # must be possible for robot to get on
    
    # Set the spawn area to a wide, flat floor
    height_field[:spawn_length, :] = 0

    # Put initial goal at the end of the spawn area, centered
    cur_x = spawn_length
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    for i in range(n_beams):
        # Each beam is left- or right-shifted up to half of lateral_range
        if i % 2 == 0:
            y_shift = int(np.round((0.18 + random.uniform(0, 1) * (lateral_range/2)) / field_resolution))
            beam_mid_y = int(np.clip(mid_y - y_shift, m_to_idx(beam_width // 2), terrain_width - m_to_idx(beam_width // 2)))
        else:
            y_shift = int(np.round((0.18 + random.uniform(0, 1) * (lateral_range/2)) / field_resolution))
            beam_mid_y = int(np.clip(mid_y + y_shift, m_to_idx(beam_width // 2), terrain_width - m_to_idx(beam_width // 2)))

        # Set beam coordinates
        beam_x1 = int(cur_x)
        beam_x2 = int(np.clip(cur_x + m_to_idx(beam_length), 0, terrain_length))
        beam_y1 = int(np.clip(beam_mid_y - m_to_idx(beam_width/2), 0, terrain_width))
        beam_y2 = int(np.clip(beam_mid_y + m_to_idx(beam_width/2), 0, terrain_width))

        # Set the beam area to the beam height
        height_field[beam_x1:beam_x2, beam_y1:beam_y2] = beam_height

        # Set all other area in this segment to a pit (negative height)
        if i == 0:
            pit_start = spawn_length
        else:
            pit_start = prev_beam_x2

        height_field[pit_start:beam_x2, :beam_y1] = -0.8 - 0.4 * difficulty  # left pit
        height_field[pit_start:beam_x2, beam_y2:] = -0.8 - 0.4 * difficulty  # right pit

        # Set the goal at the center of the beam, 60% through its length
        goal_x = int(beam_x1 + 0.6 * (beam_x2 - beam_x1))
        goal_y = int(beam_mid_y)
        goals[i+1] = [goal_x, goal_y]

        # Advance to next beam (add gap)
        prev_beam_x2 = beam_x2
        cur_x = beam_x2 + m_to_idx(gap_length)

    # Final segment: safe ground for finish line
    safe_x1 = int(cur_x)
    safe_x2 = terrain_length
    height_field[safe_x1:safe_x2, :] = 0.0
    # Last goal: finish line at center near course end
    goals[-1] = [int((safe_x1 + safe_x2)//2), mid_y]

    return height_field, goals