import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of see-saw (tilting ramp) obstacles that test balance and adaptability: the quadruped must cross tilting ramps of varying steepness and width."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # Parameters for see-saw obstacle sequence
    mid_y = m_to_idx(width) // 2
    spawn_length = m_to_idx(2.0)
    n_seesaws = 3
    # Difficulty shapes ramp height, slope, and narrowing
    base_ramp_len = 1.8 - 0.7 * difficulty
    min_ramp_w = 1.2 - 0.6 * difficulty
    max_theta = np.deg2rad(12 + 13 * difficulty)  # Ramp max tilt angle: 12 deg (easy) to 25 deg (hard)

    gap_between = 0.6 - 0.3 * difficulty  # Smaller gaps for harder
    gap_between = max(gap_between, 0.2)
    gap_height = -0.20 - 0.5 * difficulty   # Slight pit below ramps

    start_x = spawn_length
    noise_range = m_to_idx(0.3 + 0.2 * difficulty)
    terrain_shape = height_field.shape

    cur_x = start_x
    obs_half_w = m_to_idx(min_ramp_w / 2)
    platform_height = 0  # Flat between seesaws 

    # Make sure spawn area is flat
    height_field[0:spawn_length, :] = 0
    goals[0] = [m_to_idx(1.2), mid_y]

    ramp_params = []
    # Generate seesaws
    for i in range(n_seesaws):
        ramp_len = m_to_idx(base_ramp_len + random.uniform(-0.1, 0.1))
        ramp_w = m_to_idx(min_ramp_w + random.uniform(0, 0.1))
        half_w = ramp_w // 2
        theta = random.uniform(0.7, 1.0) * max_theta * ((-1) ** i)  # Alternate ramp up/down
        pivot_rel = random.uniform(0.45, 0.55)   # Pivot point is near ramp center

        # Center the ramp in y, allow some y jitter for variety
        ramp_center_y = mid_y + random.randint(-noise_range, noise_range)
        ramp_y1 = max(ramp_center_y - half_w, 0)
        ramp_y2 = min(ramp_center_y + half_w, terrain_shape[1])

        ramp_x1 = cur_x
        ramp_x2 = cur_x + ramp_len

        # Calculate ramp profile
        pivot_x = int(ramp_x1 + pivot_rel * (ramp_x2 - ramp_x1))
        pre_len = pivot_x - ramp_x1
        post_len = ramp_x2 - pivot_x

        # Left side slopes up, right side slopes down (see-saw structure)
        for x in range(ramp_x1, ramp_x2):
            if x < pivot_x:
                z = platform_height + np.tan(theta) * ((x - ramp_x1) * field_resolution)
            else:
                z = platform_height + np.tan(-theta) * ((x - pivot_x) * field_resolution)
            height_field[x, ramp_y1:ramp_y2] = z
        ramp_params.append((ramp_x1, ramp_x2, ramp_y1, ramp_y2, pivot_x, theta))

        # Make a pit between seesaws (forces to get on ramp, and punishes falling)
        pit_x1 = ramp_x2
        pit_x2 = int(ramp_x2 + m_to_idx(gap_between))
        height_field[pit_x1:pit_x2, :] = gap_height

        # Next X location
        cur_x = pit_x2
        platform_height = 0  # reset

        # Place a goal just after each seesaw's end
        if i < 2:
            g_y = min(max(ramp_center_y, m_to_idx(0.7)), terrain_shape[1] - m_to_idx(0.7))
            goals[i+1] = [min(ramp_x2 + m_to_idx(0.2), terrain_shape[0]-1), g_y]
        else:
            # Last seesaw, goal on end
            goals[i+1] = [min(ramp_x2 + m_to_idx(0.2), terrain_shape[0]-1), ramp_center_y]

    # Fill the last stretch with a flat, accessible platform leading to final goal
    last_platform_x1 = min(cur_x, terrain_shape[0])
    height_field[last_platform_x1:, :] = 0

    # Make the final goal at the end of the course
    goals[4] = [terrain_shape[0] - m_to_idx(0.6), mid_y]

    return height_field, goals