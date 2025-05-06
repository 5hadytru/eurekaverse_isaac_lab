import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of dog park-style hurdles: adjustable spaced jump bars of varying heights, requiring the quadruped to run, jump, and land repeatedly."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for hurdle sizing
    hurdle_width = 1.4  # ensures at least 1m wide
    hurdle_thickness = 0.08  # ~8cm thick bar
    # Hurdle height increases with difficulty (range: 0.10–0.35m)
    min_hurdle_height = 0.10 + 0.10 * difficulty
    max_hurdle_height = 0.16 + 0.19 * difficulty

    # Horizontal space between hurdles (meters)
    base_gap = 1.1 + 1.8 * (1-difficulty)  # closer for hard, further for easy
    hurdle_count = 6  # leave room for 2 plain goals

    spawn_x = m_to_idx(1.0)
    course_start = m_to_idx(2.0)  # No obstacles within first 2m

    course_end = m_to_idx(length-0.5)
    mid_y = m_to_idx(width/2)
    hurdle_width_idx = m_to_idx(hurdle_width)
    hurdle_half_width_idx = hurdle_width_idx // 2
    hurdle_thickness_idx = max(1, m_to_idx(hurdle_thickness))  # ensure at least 1 idx

    # Floor is flat except for the hurdles (bars)
    height_field[:,:] = 0

    # Place first goal at spawn
    goals[0] = [spawn_x, mid_y]

    # Hurdle placement
    hurdle_positions_x = []
    gap = base_gap - 0.7 * difficulty  # Pack tighter on higher difficulty

    # Calculate start positions for hurdles so that they fit nicely in terrain
    total_gap = gap * (hurdle_count)
    hurdle_start_x = course_start + m_to_idx(0.5)  # Offset half a meter past spawn area
    last_x = hurdle_start_x

    for i in range(hurdle_count):
        # Bar y-centers with some minor lateral offset for realism
        y_shift = m_to_idx(random.uniform(-0.15, 0.15) * (1-difficulty))  # Less randomness at higher difficulty
        x = int(last_x)
        y_c = mid_y + y_shift

        # Bar length (width) with margin to boundaries
        y1 = max(0, y_c-hurdle_half_width_idx)
        y2 = min(m_to_idx(width), y_c+hurdle_half_width_idx)

        # Random height between min and max
        height = random.uniform(min_hurdle_height, max_hurdle_height)

        # "Bar" shape (bar at a fixed x, with finite thickness in the x axis—else it would be infinitely thin)
        # Increase bar thickness at lower difficulty
        thickness_this = hurdle_thickness_idx + int((1-difficulty)*2)
        height_field[x:x+thickness_this, y1:y2] = height

        # Save hurdle center position for goal (centered in hurdle x, left-right middle)
        hurdle_positions_x.append(x + thickness_this//2)
        last_x = x + m_to_idx(gap + 0.45 * random.uniform(-1, 1) * (1-difficulty))  # add more randomness for easy courses

    # Fill goals: first at spawn, then right before each hurdle, then after last hurdle, and finish.
    # We want the robot to approach each hurdle straight, so place the goals at clear spots.
    for i in range(hurdle_count):
        # A goal just before each hurdle
        xg = hurdle_positions_x[i] - m_to_idx(0.25)
        xg = max(course_start, xg)  # stay in bounds
        goals[i+1] = [xg, mid_y]

    # After last hurdle, add goal on the "landing pad"
    last_landing_x = min(hurdle_positions_x[-1] + m_to_idx(0.6), course_end)
    goals[-1] = [last_landing_x, mid_y]

    return height_field, goals