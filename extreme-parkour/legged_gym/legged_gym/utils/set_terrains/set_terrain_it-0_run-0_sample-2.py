import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Set of increasingly tall stairs, testing the quadruped's ability to climb and descend realistic step heights."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course parameters
    stairwell_start_x = m_to_idx(2.0)  # Keep 2 meters clear for spawn
    stairwell_end_x = m_to_idx(length - 2.0)
    mid_y = m_to_idx(width / 2)
    stairway_width = m_to_idx(1.2)  # Wider than robot for natural stairwell
    stairway_y1 = mid_y - stairway_width // 2
    stairway_y2 = mid_y + stairway_width // 2

    num_steps = 7
    step_length_min, step_length_max = 0.6, 1.1  # Meters
    step_lengths = np.linspace(
        step_length_min, step_length_max, num_steps
    )

    # Step heights increase with difficulty
    base_height = 0.04
    max_height = 0.28 + 0.18 * difficulty  # Final step near max climbable for robot
    step_heights = np.linspace(base_height, max_height, num_steps)

    # Place first flat entry region
    height_field[:stairwell_start_x, :] = 0  # entry area
    goals[0] = [m_to_idx(0.9), mid_y]  # Start goal before stairs

    # Place ascending stairs
    cur_x = stairwell_start_x
    for i in range(num_steps):
        step_len = m_to_idx(step_lengths[i])
        h = step_heights[i]
        # Place the step (raise only in stairwell, rest remains at height 0 for safety)
        height_field[cur_x:cur_x+step_len, stairway_y1:stairway_y2] = h
        # Short vertical drop-offs at the stairwell's sides ("curbs") to encourage alignment
        height_field[cur_x:cur_x+step_len, :stairway_y1] = -0.20
        height_field[cur_x:cur_x+step_len, stairway_y2:] = -0.20
        # Place a goal near the center of each step
        center_x = cur_x + step_len // 2
        goals[i+1] = [center_x, mid_y]
        cur_x += step_len

    # Place a long plateau at the top for goal 7:
    plateau_len = m_to_idx(1.2)
    height_field[cur_x:cur_x+plateau_len, stairway_y1:stairway_y2] = max_height
    height_field[cur_x:cur_x+plateau_len, :stairway_y1] = -0.20
    height_field[cur_x:cur_x+plateau_len, stairway_y2:] = -0.20
    goals[7] = [cur_x + plateau_len // 2, mid_y]
    cur_x += plateau_len

    # End region: descend to ground level, but let the robot pause at the finish
    height_field[cur_x:, :] = 0

    return height_field, goals