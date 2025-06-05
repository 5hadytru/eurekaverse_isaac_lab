import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of low, narrow balance beams crossing a shallow trench, testing precise paw placement and balancing."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # Set spawn area to flat ground, always at z=0
    spawn_length = m_to_idx(2)
    height_field[:spawn_length, :] = 0

    # The trench: a "pit" that runs through the course, except for the start and end
    trench_depth = -0.19 - 0.21 * difficulty   # Up to -0.4m at maximum difficulty
    trench_start_x = spawn_length
    trench_end_x = m_to_idx(length-1)
    height_field[trench_start_x:trench_end_x, :] = trench_depth

    # Plan the number and position of the balance beams
    n_beams = 4
    # Make sure beam + gap fits into trench (leave final ~1m for flat ground)
    beam_total_zone = length - 3    # subtract spawn + landing length
    beam_zone_per_beam = beam_total_zone / n_beams

    # Beams parameters
    beam_length = 0.8 + 0.8 * (1 - difficulty)     # Shorter beams at higher difficulty (min 0.8m, max 1.6m)
    min_beam_width = 0.13 + 0.07 * (1 - difficulty) # Narrower beams at higher difficulty (min 0.13m, max 0.2m)
    beam_height = 0.11 + 0.12 * difficulty         # Beams just above trench at low diff, to 0.23m above at hard

    # Place beams, interspersed with random gaps
    mid_y = m_to_idx(width/2)
    current_x = trench_start_x
    beam_indices = []
    for b in range(n_beams):
        # Offset center of beam slightly to meander path
        c_y = mid_y + random.randint(-m_to_idx(0.5), m_to_idx(0.5))
        # Beam length and width (randomize a bit for realism)
        blen = m_to_idx(beam_length + random.uniform(-0.1, 0.1))
        bwidth = m_to_idx(min_beam_width + random.uniform(0, 0.09))
        half_w = bwidth//2

        # Choose start/end for this beam segment
        beam_start_x = current_x + m_to_idx(0.1)   # small buffer
        beam_end_x = min(beam_start_x + blen, trench_end_x - m_to_idx((n_beams-b-1)*beam_zone_per_beam))
        y1 = max(0, c_y - half_w)
        y2 = min(m_to_idx(width), c_y + half_w)

        # Place the beam
        height_field[beam_start_x:beam_end_x, y1:y2] = beam_height

        # Remember the beam's center for goal placement
        beam_center_x = int((beam_start_x + beam_end_x) / 2)
        beam_center_y = int((y1 + y2) / 2)
        beam_indices.append((beam_center_x, beam_center_y))

        # Advance to next zone, leaving a gap (pit) between beams
        gap_len = m_to_idx(beam_zone_per_beam) - blen
        gap = max(m_to_idx(0.45), min(m_to_idx(1.6), gap_len + random.randint(-m_to_idx(0.1), m_to_idx(0.1))))
        current_x = int(beam_end_x + gap)
        if current_x > trench_end_x - m_to_idx(beam_zone_per_beam):
            break  # End of trench

    # Landing area at end (flat ground)
    landing_x_start = min(current_x, m_to_idx(length-1))
    height_field[landing_x_start:, :] = 0

    # Set goals:
    # 0: spawn
    goals[0] = [m_to_idx(1.0), m_to_idx(width/2)]
    # 1-4: center of each beam
    for i, (cx, cy) in enumerate(beam_indices[:4]):
        goals[i+1] = [cx, cy]

    return height_field, goals