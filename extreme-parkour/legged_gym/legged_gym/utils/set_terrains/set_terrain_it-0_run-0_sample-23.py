import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of staggered narrow beams above pits to test precise foot placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- Terrain & beam specifications ---
    # Beams are narrow: width = 0.4 to 0.7m (harder for higher difficulty) 
    # Beams are at similar height, but always above a -1m pit.
    # Height of beams increases with difficulty to require precise stepping.
    # Beams are staggered to require both turning and careful correction.

    min_beam_width = 0.7 - 0.3 * difficulty   # Between 0.7m (easy) and 0.4m (hard)
    min_beam_width = max(0.4, min_beam_width)
    max_beam_width = min_beam_width + 0.2     # beams not usually much wider
    min_beam_len  = 2.4                       # at least 2 meters per beam
    max_beam_len  = 2.8                       # a little more, for stagger effect
    beam_height   = 0.07 + 0.18 * difficulty  # 7cm (easy) to 25cm (hard), above zero

    n_beams = 6
    spawn_area_x = m_to_idx(2)
    total_length = m_to_idx(length)
    total_width  = m_to_idx(width)

    # --- Prepare the pit area (all pit = -1.0, except spawn and last area) ---
    height_field[:,:] = -1.0
    # Set spawn region flat
    height_field[0:spawn_area_x, :] = 0.0
    # Set final meter flat (used for final goal)
    height_field[-m_to_idx(1):, :] = 0.0

    # --- Beam configuration ---
    # Beams are staggered left/right. Each requires a ~30-45 degree turn from the previous beam.
    # Each goal is centered on a beam.
    mid_y = total_width // 2
    beam_centers = []  # each as (x, y), for goals

    cur_x = spawn_area_x
    last_y = mid_y
    turn_directions = [1, -1] * 4  # alternate left/right

    for beam_i in range(n_beams):
        beam_len  = m_to_idx(random.uniform(min_beam_len, max_beam_len))
        beam_width = m_to_idx(random.uniform(min_beam_width, max_beam_width))

        # The amount to stagger sideways, as a function of difficulty
        y_stagger = m_to_idx((0.6 + 0.9 * difficulty) * turn_directions[beam_i])
        candidate_y = last_y + y_stagger

        # Clamp beam to within field
        y0 = max(0, min(total_width - beam_width, candidate_y - beam_width // 2))
        y1 = y0 + beam_width
        x0 = cur_x
        x1 = min(x0 + beam_len, total_length - m_to_idx(3))

        # Raise beam above pit
        height_field[x0:x1, y0:y1] = beam_height

        # Place goal at center of beam
        xg = (x0 + x1) // 2
        yg = (y0 + y1) // 2
        beam_centers.append((xg, yg))
        last_y = yg

        # Move x forward for next beam
        gap_len = m_to_idx(0.6 + 0.5 * difficulty)  # bigger gap for higher difficulty
        cur_x = x1 + gap_len

    # --- Place goals ---
    # 0th goal in spawn area, at spawn line
    goals[0] = [spawn_area_x - m_to_idx(0.7), mid_y]
    for i in range(n_beams):
        goals[i+1] = list(beam_centers[i])
    # Last goal on finish line, after last beam
    final_x = total_length - m_to_idx(0.7)
    goals[-1] = [final_x, last_y]

    # --- Ensure last region is flat ground (at 0) for robot to reach the last goal ---
    height_field[final_x-m_to_idx(0.5):, :] = 0.0

    return height_field, goals