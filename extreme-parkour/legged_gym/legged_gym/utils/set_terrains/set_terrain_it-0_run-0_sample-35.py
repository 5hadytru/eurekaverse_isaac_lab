import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of adjustable-width balance beams spanning pits, focusing on testing the quadruped's balance and precise foot placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters (scaled with difficulty)
    num_beams = 5
    beam_length = 1.5                                 # meters
    min_beam_width = 0.4 + (0.4 * (1-difficulty))     # meters, narrower as difficulty increases
    max_beam_width = 0.7 - (0.3 * difficulty)         # meters, upper bound for width, wider at low difficulty
    min_gap = 0.6 + 0.6 * difficulty                  # meters, longer gaps at high difficulty
    pit_depth = 0.7 + 0.5 * difficulty                # meters
    beam_height = 0.07 + 0.10 * difficulty            # slightly raised at high difficulty (need precise placements)
    center_y = m_to_idx(width // 2)
    
    # Wide enough spawn region before obstacles (avoid spawn overlap)
    spawn_x = m_to_idx(1.0)
    height_field[:spawn_x, :] = 0                     # flat ground for spawn region
    goals[0] = [spawn_x - m_to_idx(0.5), center_y]    # goal just before first pit

    # Mark pits after spawn region
    x_cursor = m_to_idx(2.0)
    for b in range(num_beams):
        beam_w = m_to_idx(random.uniform(min_beam_width, max_beam_width))
        gap = m_to_idx(min_gap)                       # fixed pit gap between beams
        
        # Pit region
        pit_start = x_cursor - gap//2
        pit_end = x_cursor + m_to_idx(beam_length) + gap//2
        pit_start  = max(pit_start, spawn_x)
        pit_end = min(pit_end, m_to_idx(length))
        height_field[pit_start:pit_end, :] = -pit_depth

        # Add balance beam in center of course
        beam_start = x_cursor
        beam_end = beam_start + m_to_idx(beam_length)
        beam_y1 = center_y - beam_w//2
        beam_y2 = center_y + beam_w//2
        beam_y1 = max(beam_y1, 0)
        beam_y2 = min(beam_y2, m_to_idx(width))
        height_field[beam_start:beam_end, beam_y1:beam_y2] = beam_height

        # Place goal at center of this beam
        goals[b+1] = [beam_start + (beam_end-beam_start)//2, center_y]

        # Stagger beams left/right (slightly), as difficulty increases
        if difficulty > 0.3 and b > 0:
            y_shift = m_to_idx(random.uniform(-0.8*difficulty, 0.8*difficulty))
            center_y = int(np.clip(center_y + y_shift, m_to_idx(0.6), m_to_idx(width-0.6)))
        
        # Move cursor past this beam and pit for the next
        x_cursor = beam_end + gap

    # Land zone: flat ground after final beam for recovery
    land_start = int(min(x_cursor, m_to_idx(length)))
    height_field[land_start:, :] = 0

    # Place final goal at the end of the course
    goals[7] = [m_to_idx(length-0.5), center_y]

    # If fewer than 8 obstacles, fill the unused goals with the endpoint
    for i in range(num_beams+2, 8):
        goals[i] = [m_to_idx(length-0.5), center_y]

    return height_field, goals