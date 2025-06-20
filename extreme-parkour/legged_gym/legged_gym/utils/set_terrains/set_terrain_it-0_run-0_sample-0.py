import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Alternating series of ascending and descending sloped ramps ('A-frames') with flat tops, testing dynamic balance on inclines and declines."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # Course Parameters
    ramp_base_length = 1.2 - 0.4 * difficulty    # meters, gets shorter with difficulty
    ramp_top_length = 0.6 - 0.2 * difficulty     # meters, gets shorter with difficulty
    ramp_total_length = 2 * ramp_base_length + ramp_top_length
    ramp_width = 1.2 + 0.7 * (1 - difficulty)    # ramps are narrower at high difficulty
    ramp_height = 0.13 + 0.22 * difficulty       # A-frame gets much steeper higher difficulty

    n_ramps = 4
    pit_depth = -0.15 - 0.25 * difficulty
    pit_length = 0.6 + 0.6 * difficulty

    # Convert to indices
    ramp_base_length = m_to_idx(ramp_base_length)
    ramp_top_length = m_to_idx(ramp_top_length)
    ramp_width = m_to_idx(ramp_width)
    ramp_total_length = 2*ramp_base_length + ramp_top_length
    ramp_height = float(ramp_height)
    pit_length = m_to_idx(pit_length)

    # Find midline for ramp y-placement
    y_mid = m_to_idx(width // 2)

    def add_ramp(x_start, direction=1):
        """Draw an A-frame ramp with direction=1 (up), direction=-1 (down)"""
        x0 = x_start
        x1 = x0 + ramp_base_length
        x2 = x1 + ramp_top_length
        x3 = x2 + ramp_base_length

        y0 = max(0, y_mid - ramp_width // 2)
        y1 = min(m_to_idx(width), y_mid + ramp_width // 2)
        # Slope up
        for xi in range(x0, x1):
            p = (xi - x0) / max(1, (x1 - x0))
            if direction == 1:
                val = ramp_height * p
            else:
                val = ramp_height * (1 - p)
            height_field[xi, y0:y1] = val
        # Flat top
        for xi in range(x1, x2):
            height_field[xi, y0:y1] = ramp_height if direction == 1 else 0
        # Slope down
        for xi in range(x2, x3):
            p = (xi - x2) / max(1, (x3 - x2))
            if direction == 1:
                val = ramp_height * (1 - p)
            else:
                val = ramp_height * p
            height_field[xi, y0:y1] = val
        return (x0, x1, x2, x3), (y0, y1)
    
    # Course starts with a flat safe zone
    spawn_x = m_to_idx(2)
    height_field[:spawn_x, :] = 0
    goals[0] = [spawn_x - m_to_idx(0.5), y_mid]

    current_x = spawn_x
    direction = 1  # First ramp goes up
    for i in range(n_ramps):
        # Pit between ramps
        if i > 0:
            height_field[current_x:current_x + pit_length, :] = pit_depth
            current_x = current_x + pit_length

        # Add ramp
        (x0, x1, x2, x3), (y0, y1) = add_ramp(current_x, direction=direction)
        # Place goal at the center of the flat top
        goal_x = (x1 + x2) // 2
        goal_y = (y0 + y1) // 2
        goals[i + 1] = [goal_x, goal_y]
        current_x = x3

        # Alternate ramp facing direction (A-frames, up then down)
        direction *= -1

    # Flat finish section
    height_field[current_x:, :] = 0

    # Ensure all goals are within bounds
    for j in range(5):
        gx, gy = goals[j]
        gx, gy = int(np.clip(gx, 0, height_field.shape[0]-1)), int(np.clip(gy, 0, height_field.shape[1]-1))
        goals[j] = [gx, gy]

    return height_field, goals