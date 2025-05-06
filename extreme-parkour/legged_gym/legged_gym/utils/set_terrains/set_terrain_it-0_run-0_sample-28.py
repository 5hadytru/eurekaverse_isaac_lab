import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping-stone balance beams: narrow beams of varying widths over a pit to test balance and precise foot placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters
    mid_y = m_to_idx(width) // 2
    spawn_length = m_to_idx(2)
    total_length = m_to_idx(length)
    total_width = m_to_idx(width)

    # Pit setup
    # Make ground flat up to spawn region
    height_field[:spawn_length, :] = 0

    # Everything after spawn is a pit of -0.85 m
    height_field[spawn_length:, :] = -0.85

    # Beam parameters based on difficulty
    num_beams = 6
    base_beam_length = 1.4
    base_beam_width = 1.1 - 0.5 * difficulty   # 1.1 m (easy) to 0.6 m (hard)
    base_beam_height = 0.0   # Top flush with spawn

    # The robot is 0.28 m wide; at hardest, beam is twice that (0.6 - enough for challenge, not frustration)
    beam_lengths = []
    beam_widths = []
    beam_offsets = []
    dx_between = (length - 2.8 - num_beams * base_beam_length) / (num_beams + 1)
    dx_between = max(dx_between, 0.18)  # Ensure minimal separation possible

    cur_x = float(2.0)
    beam_centers_x = []
    beam_centers_y = []

    for i in range(num_beams):  # Place beams
        # Random offset in y for each beam (snake/wave pattern to force minor turning or repositioning)
        y_offset = random.uniform(-0.5, 0.5) * (0.5 + 0.9 * difficulty)  # wider variation as diff ↑
        x1 = cur_x
        x2 = cur_x + base_beam_length
        y_center = (width / 2) + y_offset
        y1 = y_center - base_beam_width / 2
        y2 = y_center + base_beam_width / 2

        # Convert to indices; clip to bounds
        x1i, x2i = max(m_to_idx(x1), spawn_length), min(m_to_idx(x2), total_length)
        y1i, y2i = max(m_to_idx(y1), 0), min(m_to_idx(y2), total_width)
        # Draw beam: top surface flush with spawn, pit below
        height_field[x1i:x2i, y1i:y2i] = base_beam_height

        # Store for goal position
        beam_centers_x.append((x1i + x2i) // 2)
        beam_centers_y.append((y1i + y2i) // 2)
        # Next beam
        cur_x += base_beam_length + dx_between + random.uniform(-0.1, 0.1) # slight jitter

    # Entry and exit ramp, and final flat ground at end
    # Ramps are not required; robot climbs on/off from ground level.

    # Set Goals: 
    # 0 - Spawn point right before pit
    goals[0] = [m_to_idx(1.6), mid_y]
    # 1-6: Center of each beam
    for i in range(num_beams):
        goals[i+1, 0] = beam_centers_x[i]
        goals[i+1, 1] = beam_centers_y[i]
    # 7: Exit goal at end, back in flat ground (map x = length-0.7, center y)
    x_exit = m_to_idx(length - 0.7)
    goals[7] = [x_exit, mid_y]
    height_field[x_exit:, :] = 0  # Put exit ground at height 0

    return height_field, goals