import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of ascending and descending ramps to test climbing and descending agility."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]
    
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # DESIGN OF THE OBSTACLE COURSE:
    # 8 ramps, alternating ascending and descending
    # Each ramp is 1.2m long and 1.5m wide, with a 0.2m flat "rest" area in between.
    # Ramp heights vary based on difficulty, so the max ramp angle increases as difficulty increases.
    # Begin course after the first 2m to allow spawning on flat ground.
    # Robot travels in a gentle S-curve along y as it proceeds, forcing light turning.

    # CONSTANTS
    n_ramps = 8
    start_x = 2.0  # meters (spawn buffer)
    total_ramp_l = 1.2 * n_ramps + 0.2 * (n_ramps-1)
    end_x = min(start_x + total_ramp_l, length-0.5)
    ramp_length = 1.2
    ramp_width = 1.5
    rest_length = 0.2
    min_height = 0.07      # Minimum height per ramp
    max_height = 0.35      # Maximum height per ramp at difficulty==1
    
    mid_y = m_to_idx(width) // 2
    swing = m_to_idx((width-ramp_width-0.3)/2)  # max deviation for S-curve

    # Flat spawn area
    height_field[:m_to_idx(2), :] = 0
    goals[0] = [m_to_idx(1), mid_y]

    x = start_x
    last_top = 0
    for i in range(n_ramps):
        # Alternating: up, down, up, ...
        is_ascend = (i % 2 == 0)
        amplitude = min_height + (max_height-min_height) * difficulty
        height_change = (amplitude if is_ascend else -amplitude)
        next_top = last_top + height_change

        # S-curve: slight lateral shift of ramp over course
        s_frac = (i/(n_ramps-1))*2-1  # goes from -1 to +1 over the ramps
        y_center = mid_y + int(swing * 0.8 * np.sin(s_frac * np.pi/2))
        y1 = max(0, y_center - m_to_idx(ramp_width/2))
        y2 = min(m_to_idx(width), y_center + m_to_idx(ramp_width/2))

        # Define ramp region
        x1 = m_to_idx(x)
        x2 = m_to_idx(x + ramp_length)

        # Linear height profile along ramp
        for ramp_idx, xi in enumerate(range(x1, x2)):
            frac = ramp_idx / max(1, x2-x1-1)
            h = last_top + frac * (next_top - last_top)
            height_field[xi, y1:y2] = h

        # Rest/flat area after ramp
        rx1 = x2
        rx2 = min(m_to_idx(x + ramp_length + rest_length), m_to_idx(length))
        height_field[rx1:rx2, y1:y2] = next_top

        # Place goal at end of ramp, slightly into the flat spot for safety
        goal_x = x + ramp_length + 0.1
        goals[i+1] = [m_to_idx(goal_x), y_center]

        # Update position
        x = x + ramp_length + rest_length
        last_top = next_top

    # Fill the final area (after last ramp) with the final top height
    if x < length:
        height_field[m_to_idx(x):, :] = last_top

    # Edge guards: ensure full ramp width fits in field
    height_field[:, :m_to_idx(0.07)] = last_top  # Left edge guard
    height_field[:, -m_to_idx(0.07):] = last_top # Right edge guard

    return height_field, goals