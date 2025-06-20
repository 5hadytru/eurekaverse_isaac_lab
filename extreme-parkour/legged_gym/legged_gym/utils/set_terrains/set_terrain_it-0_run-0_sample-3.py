import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A balance beam course: tests precise foot placement and lateral balance by requiring the quadruped to traverse a series of progressively narrower, elevated beams with small connecting flat landings."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        if isinstance(m, list) or isinstance(m, tuple):
            return [round(i / field_resolution) for i in m]
        return np.round(m / field_resolution).astype(np.int16)

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))
    
    # PARAMETERS
    # Beam length and base height
    min_beam_length, max_beam_length = 1.8, 2.4
    base_height = 0.20 + 0.10 * difficulty      # base elevation
    pit_depth = -0.7 - 0.6 * difficulty         # depth of off-beam areas
    # Narrower beams at higher difficulty
    min_beam_width = 1.0 - 0.65 * difficulty    # from 1.0m (easy) down to 0.35m (hard)
    min_beam_width = max(min_beam_width, 0.35)  # do not allow narrow beams < 0.35m

    spawn_length = m_to_idx(2.0)                # leave 2m spawn area
    field_len = m_to_idx(length)
    field_wid = m_to_idx(width)
    mid_y = field_wid // 2

    # Make starting area flat and level
    height_field[:spawn_length, :] = 0.0
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    num_beams = 4                              # four beams, with last segment as a wide finish pad
    gap_length = 0.5 + 0.5 * difficulty         # gaps between beams become longer with difficulty
    gap_length_idx = m_to_idx(gap_length)

    beam_starts = [spawn_length + i * (int(np.mean([m_to_idx(min_beam_length), m_to_idx(max_beam_length)]) + gap_length_idx)) for i in range(num_beams)]
    # Some small random shift (sideways) for later beams
    max_lateral_shift = m_to_idx(0.4 + 0.4*difficulty)

    curr_x = spawn_length
    for i in range(num_beams):
        # Define beam geometry
        beam_length = m_to_idx(random.uniform(min_beam_length, max_beam_length))
        # Center all beams but progressively add random lateral offset to make them misaligned
        if i == 0:
            beam_center_y = mid_y
        else:
            beam_center_y += random.randint(-max_lateral_shift, max_lateral_shift)
            beam_center_y = max(m_to_idx(min_beam_width)//2, min(field_wid - m_to_idx(min_beam_width)//2, beam_center_y))

        beam_half_width = m_to_idx(min_beam_width / 2.0)
        
        # Build beam (it's like a raised platform)
        # All other areas beside beam are pits.
        x1, x2 = curr_x, min(curr_x + beam_length, field_len)
        y1, y2 = beam_center_y - beam_half_width, beam_center_y + beam_half_width
        y1 = max(0, y1)
        y2 = min(field_wid, y2)

        height_field[x1:x2, :] = pit_depth          # pit outside the beam
        height_field[x1:x2, y1:y2] = base_height    # beam itself

        # Flat "connectors" between beams as landings (for goals)
        connector_length = m_to_idx(0.4)
        # At the start of first beam, keep a short flat area
        if i == 0:
            height_field[x1-connector_length:x1, :] = 0.0
        # At the end of every beam, add landing pad (wider)
        landing_y1 = max(0, beam_center_y - m_to_idx(0.7))
        landing_y2 = min(field_wid, beam_center_y + m_to_idx(0.7))
        height_field[x2:x2+connector_length, landing_y1:landing_y2] = 0.0

        # Save goal: set at the landing after each beam
        goal_x = int(x2 + connector_length // 2)
        goal_y = beam_center_y
        if i < 4:
            goals[i+1] = [goal_x, goal_y]

        # Move for next beam (gap in terrain)
        curr_x = int(x2 + connector_length + gap_length_idx)

    # Make rest of field flat as a finish zone
    height_field[curr_x:, :] = 0.0

    # Clamp all goal indices to the map
    for i in range(goals.shape[0]):
        goals[i, 0] = np.clip(goals[i, 0], 0, field_len-1)
        goals[i, 1] = np.clip(goals[i, 1], 0, field_wid-1)

    return height_field, goals