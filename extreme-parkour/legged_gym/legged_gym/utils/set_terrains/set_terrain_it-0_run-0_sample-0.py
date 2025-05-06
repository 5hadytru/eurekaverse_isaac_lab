import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of angled ramps for quadruped stair/ramp ascent and descent skill."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for ramps
    # Ramp configuration scales with difficulty to increase the steepness and height
    num_ramps = 4
    # Ramp length: 1.8m ~ 1.0m as difficulty increases
    ramp_length = np.linspace(1.8, 1.0, num=num_ramps) * (1 - 0.5 * difficulty)
    # Ramp width: always wide, but shift slightly side-to-side
    ramp_width = 1.2
    min_side_margin = 0.4
    mid_y = m_to_idx(width / 2)
    ramp_width_idx = m_to_idx(ramp_width // 2)
    ramp_spacing = 0.3 + 1.0 * difficulty  # flat in-between ramps

    # Ramp ascent/descent height increases with difficulty
    ramp_height = 0.12 + 0.23 * difficulty
    flat_section_length = 0.7 - 0.2 * difficulty
    flat_section_length = max(flat_section_length, 0.25)

    # Ensure start flat area for spawn
    spawn_length = m_to_idx(2)
    height_field[0:spawn_length, :] = 0
    goals[0] = [m_to_idx(1.0), mid_y]

    cur_x = spawn_length
    ramp_directions = [1, -1] * (num_ramps // 2)  # Alternate up and down
    y_shift_options = [-0.2, 0.0, 0.2]

    for i in range(num_ramps):
        # Ramp placement
        r_len = m_to_idx(ramp_length[i])
        f_len = m_to_idx(flat_section_length)
        y_shift = m_to_idx(random.choice(y_shift_options))
        y_mid = int(np.clip(mid_y + y_shift, m_to_idx(min_side_margin), m_to_idx(width) - m_to_idx(min_side_margin)))

        # Ramp indices
        x1 = cur_x
        x2 = x1 + r_len
        y1 = y_mid - ramp_width_idx
        y2 = y_mid + ramp_width_idx
        y1 = max(y1, 0)
        y2 = min(y2, m_to_idx(width))

        # Ramp direction: up or down
        dir = ramp_directions[i]
        start_h = np.max(height_field[x1-1, y1:y2]) if x1 > 0 else 0.0
        end_h = start_h + ramp_height * dir

        # Create ramp: linear slope along x
        for x in range(x1, x2):
            t = (x - x1) / max(1, r_len - 1)
            height_field[x, y1:y2] = (1-t)*start_h + t*end_h

        # Flat section at top/bottom
        x_flat1 = x2
        x_flat2 = x_flat1 + f_len
        height_field[x_flat1:x_flat2, y1:y2] = end_h

        # Place goal at middle of each flat section
        goal_x = int(x_flat1 + (x_flat2 - x_flat1)//2)
        if i+1 < 8:
            goals[i+1] = [goal_x, y_mid]

        # Update for next ramp
        cur_x = x_flat2 + m_to_idx(ramp_spacing)

    # Final goal at end of last flat
    final_goal_x = min(cur_x, height_field.shape[0]-1)
    goals[7] = [final_goal_x, mid_y]

    # Clean up goal indices to stay in bounds
    goals = np.clip(goals, [0,0], [height_field.shape[0]-1, height_field.shape[1]-1])

    # Fill any remaining flat region to the end with last height value
    if cur_x < height_field.shape[0]:
        last_height = np.mean(height_field[cur_x-1, :])
        height_field[cur_x:, :] = last_height

    return height_field, goals