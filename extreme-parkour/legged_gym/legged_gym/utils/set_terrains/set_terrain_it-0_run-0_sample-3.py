import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Alternating angled ramps for testing balancing and ascending/descending skills."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return (np.round(m / field_resolution).astype(np.int16)
                if not (isinstance(m, list) or isinstance(m, tuple))
                else [round(i / field_resolution) for i in m])

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course parameters
    total_ramps = 6
    spawn_length = m_to_idx(2)
    length_avail = m_to_idx(length) - spawn_length
    mid_y = m_to_idx(width // 2)
    course_width = m_to_idx(width)
    safe_margin = m_to_idx(0.5)  # margin on the sides for safety

    # Ramp geometry
    ramp_min_length = 1.2      # meters
    ramp_max_length = 2.0
    ramp_length = np.linspace(
        ramp_min_length, ramp_max_length - difficulty * 0.5, total_ramps)
    ramp_length = [m_to_idx(l) for l in ramp_length]
    flat_top_length = m_to_idx(0.3 + 0.2 * (1 - difficulty))

    ramp_min_width = 1.2      # meters, always >1.0 as per spec
    ramp_max_width = min(width - 0.5, 2.0)
    ramp_w = np.linspace(ramp_min_width, ramp_max_width, total_ramps)
    ramp_widths = [m_to_idx(w) for w in ramp_w]

    # Ramp heights
    max_height = 0.25 + 0.15 * difficulty
    min_height = 0.04 + 0.1 * difficulty  # don't make too easy

    # -1 for descent, 1 for ascent
    ramp_directions = [1 if i % 2 == 0 else -1 for i in range(total_ramps)]
    # Each ramp is at a random orientation: sometimes angled forward, sometimes left/right, so need to keep the centerline zig-zag
    y_offsets = np.linspace(safe_margin, course_width - safe_margin, total_ramps + 2)[1:-1]
    y_offsets = [int(o) for o in y_offsets]

    cur_x = spawn_length

    # Set spawn area to flat ground.
    height_field[:spawn_length, :] = 0
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    for i in range(total_ramps):
        # Ramps alternate in ascent/descent
        direction = ramp_directions[i]
        length = ramp_length[i]
        width_ = ramp_widths[i]

        # Height from base to top of ramp
        ramp_height = np.random.uniform(min_height, max_height) * direction

        # Center the ramp along the y-axis, but shift to zig-zag y location.
        ramp_mid_y = y_offsets[i]

        x0 = cur_x
        x1 = cur_x + length
        y0 = max(ramp_mid_y - width_ // 2, 0)
        y1 = min(ramp_mid_y + width_ // 2, course_width)

        # Ramp slope (linear interpolation between end points)
        for xi in range(x0, x1):
            frac = (xi - x0) / max(1, x1 - x0 - 1)
            h = ramp_height * frac
            height_field[xi, y0:y1] = (
                height_field[xi, y0:y1] + h
            )  # relative to starting ground (which might be nonzero)

        # Flat platform on top of ramp for stability
        xf0, xf1 = x1, min(x1 + flat_top_length, m_to_idx(length))
        top_height = ramp_height
        height_field[xf0:xf1, y0:y1] = height_field[x1 - 1, y0:y1]
        cur_x = xf1

        # Intermediate goal: always place on the middle "flat" top of ramp
        # Or if i == last, place further along the flat ground
        if i < 7:
            goal_y = int((y0 + y1) // 2)
            goal_x = int((x1 + xf0) // 2)
            goals[i + 1] = [goal_x, goal_y]

    # Final ramp-off platform and goal
    end_pad = m_to_idx(1)
    height_field[cur_x:cur_x+end_pad, :] = 0  # return to ground level
    goals[7] = [min(cur_x + end_pad // 2, m_to_idx(length) - 2), mid_y]

    # Clip all heights so the minimum is 0
    min_spawn_height = height_field[:spawn_length, :].min()
    if min_spawn_height > 0:
        height_field -= min_spawn_height

    # Ensure all goals are within bounds
    for i in range(8):
        goals[i, 0] = min(max(0, goals[i, 0]), m_to_idx(length) - 1)
        goals[i, 1] = min(max(0, goals[i, 1]), m_to_idx(width) - 1)

    return height_field, goals