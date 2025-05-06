import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """
    Series of balance beams: The course tests the quadruped's ability to precisely traverse long, narrow beams (variable difficulty), while negotiating short transitions between beams. 
    The robot must climb onto the first beam, cross several beams (raised above pits), and turn in several places.
    """

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- Parameters ---
    # Difficulty determines beam width and beam height.
    min_beam_width = 0.4
    max_beam_width = 1.2
    beam_width = max_beam_width - (max_beam_width - min_beam_width) * difficulty   # From 1.2m (easy) to 0.4m (hard)
    beam_width_idx = m_to_idx(beam_width)

    beam_height = 0.10 + 0.18 * difficulty   # Beams are higher off ground on hard settings: up to 0.28m
    pit_depth = -0.80 - 0.6 * difficulty     # Pits underneath are deeper at harder settings

    beam_length = 2.5                         # Each beam spans about 2.5m
    beam_length_idx = m_to_idx(beam_length)

    pit_length = 0.7 + 1.0 * difficulty       # Gaps between beams are longer for more difficulty
    pit_length_idx = m_to_idx(pit_length)

    min_y_margin = 0.5                        # Minimum margin to wall from centerline of beam
    usable_y = width - 2 * min_y_margin
    # We'll alternate between center, left, and right to force turning

    start_x = m_to_idx(2.0)
    spawn_length = m_to_idx(2.0)
    N_BEAMS = 5
    beam_dirs = [0.0, -0.7, 0.7, 0.0, -0.5, 0.5]  # Y deviations to force mild left/right beams

    # --- Begin construction ---

    # Region before first obstacle -- flat
    height_field[:start_x, :] = 0

    # Place first goal near start
    mid_y = m_to_idx(width / 2)
    goals[0] = [m_to_idx(1.0), mid_y]

    # Place remaining beams and pits
    x = start_x
    beam_centers_y = [width / 2]
    curr_y = width / 2

    for i in range(N_BEAMS):
        # Compute intended y position for this beam, relative to course center
        deviation = beam_dirs[i % len(beam_dirs)] * usable_y * (0.40 + 0.26 * difficulty)  # Up to ~1m left/right
        curr_y = np.clip(width / 2 + deviation, min_y_margin, width - min_y_margin)
        beam_centers_y.append(curr_y)

        # Indices for beam placement
        cx = x + beam_length_idx // 2
        cy = m_to_idx(curr_y)
        half_width = beam_width_idx // 2

        # Place pit before beam (except before first beam)
        if i > 0:
            x_pit_start = x - pit_length_idx
            height_field[x_pit_start:x, :] = pit_depth

        # Place beam (raised off pits)
        x_beam_start = x
        x_beam_end = x + beam_length_idx
        y1 = cy - half_width
        y2 = cy + half_width
        y1_clipped = max(0, y1)
        y2_clipped = min(m_to_idx(width), y2)
        height_field[x_beam_start:x_beam_end, y1_clipped:y2_clipped] = beam_height

        # Place goal at the center of each beam
        if i < 7:
            goals[i+1] = [cx, cy]

        # Next start x: move past this beam and next pit
        x = x_beam_end + pit_length_idx

    # The rest is flat ground
    height_field[x:, :] = 0
    # Place final goal near end
    if N_BEAMS + 1 < 8:
        for idx in range(N_BEAMS + 1, 8):
            # Evenly space final remaining goals to the flat end of the course
            goals[idx] = [min(m_to_idx(length)-1, x + m_to_idx(0.5 * (idx - N_BEAMS))), m_to_idx(beam_centers_y[-1])]

    return height_field, goals