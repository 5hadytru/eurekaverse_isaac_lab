import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of alternating ascending and descending ramps, each connected by a step ledge, to test the quadruped's ability to smoothly handle sloped and stepped terrain."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- Terrain parameters (all dimensions in meters unless converted) ---
    ramp_length = 1.5 - difficulty * 0.4  # ramps get shorter as difficulty increases
    ramp_height = 0.10 + difficulty * 0.26  # ramps get steeper/higher
    step_height = 0.05 + difficulty * 0.20  # steps also get steeper
    step_length = 0.4  # meters, short and abrupt
    path_width = 1.2 - 0.7 * difficulty  # minimum width = 0.5m

    pit_depth = -0.7 * difficulty  # optional: drop off on either side of path for high difficulty

    course_mid = m_to_idx(width / 2)
    path_half = m_to_idx(path_width / 2)

    spawn_length = m_to_idx(2)
    n_obstacles = 3 + int(difficulty*3)  # 3 at low diff, up to 6 at max
    x_cursor = spawn_length

    # Set spawn area to flat, full width
    height_field[:spawn_length] = 0
    goals[0] = [spawn_length // 2, course_mid]  # starting in center of flat area

    # Helper functions
    def add_ramp(start_x, end_x, y_center, width, h0, h1):
        """Adds a ramp inclined from h0 to h1."""
        y0 = y_center - width // 2
        y1 = y_center + width // 2
        n_steps = end_x - start_x
        ramp_profile = np.linspace(h0, h1, n_steps)[:, None]
        height_field[start_x:end_x, y0:y1] = ramp_profile

    def add_step(x_pos, y_center, width, height):
        """Adds a step/ledge across the path."""
        y0 = y_center - width // 2
        y1 = y_center + width // 2
        height_field[x_pos:x_pos+m_to_idx(step_length), y0:y1] = height

    # Build the obstacle course
    goal_idx = 1  # first goal after spawn
    h = 0  # ground height at the start of each segment
    for i in range(n_obstacles):
        # Ascending ramp
        x_ramp_start = x_cursor
        x_ramp_end = x_ramp_start + m_to_idx(ramp_length)
        add_ramp(x_ramp_start, x_ramp_end, course_mid, path_half*2, h, h + ramp_height)
        h += ramp_height
        x_cursor = x_ramp_end

        # Place goal at end of ramp
        if goal_idx < 8:
            goals[goal_idx] = [x_cursor - m_to_idx(0.3), course_mid]
            goal_idx += 1

        # Step ledge
        x_step = x_cursor
        step_top_h = h + step_height
        add_step(x_step, course_mid, path_half*2, step_top_h)
        x_cursor = x_step + m_to_idx(step_length)
        h = step_top_h

        # Optional: for visual structure, add pits/low ground at sides of the path for high difficulties
        if difficulty > 0.5:
            y0 = 0
            y1 = course_mid - path_half
            y2 = course_mid + path_half
            y3 = m_to_idx(width)

            height_field[(x_ramp_start):(x_cursor), y0:y1] = pit_depth
            height_field[(x_ramp_start):(x_cursor), y2:y3] = pit_depth

        # Descending ramp (for alternate segments)
        if i % 2 == 1:
            x_ramp_down_start = x_cursor
            x_ramp_down_end = x_ramp_down_start + m_to_idx(ramp_length)
            add_ramp(x_ramp_down_start, x_ramp_down_end, course_mid, path_half*2, h, h - ramp_height)
            h -= ramp_height
            x_cursor = x_ramp_down_end

            # Place goal at end of ramp
            if goal_idx < 8:
                goals[goal_idx] = [x_cursor - m_to_idx(0.3), course_mid]
                goal_idx += 1

    # Final straight/final goal
    x_final = min(m_to_idx(length)-1, x_cursor + m_to_idx(0.5))
    height_field[x_cursor:x_final, course_mid-path_half:course_mid+path_half] = h
    # Fill side pits for exit so the robot doesn’t walk off a cliff at the end
    if difficulty > 0.5:
        height_field[x_cursor:x_final, :] = h

    # Last goal (all remaining goals are set to the last position)
    for k in range(goal_idx, 8):
        goals[k] = [min(m_to_idx(length)-1, x_final), course_mid]

    return height_field, goals