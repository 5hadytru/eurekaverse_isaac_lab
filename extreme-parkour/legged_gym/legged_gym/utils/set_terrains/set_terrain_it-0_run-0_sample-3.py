import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of alternating angled ramps testing sloped walking and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Ramp parameters
    ramp_length = 1.5 - 0.3 * difficulty             # Longer ramps at easier settings
    ramp_length_i = m_to_idx(ramp_length)
    ramp_width = 1.2 - 0.4 * difficulty              # Narrower ramps at higher difficulty
    ramp_width = max(ramp_width, 0.6)
    ramp_width_i = m_to_idx(ramp_width)
    max_angle = 12 + 14 * difficulty                 # From 12 deg up to 26 deg (in degrees)
    min_angle = 7 + 3 * difficulty
    angle_rad = np.deg2rad(random.uniform(min_angle, max_angle))

    # Total ramps = 6, alternating up and down; spawn + finish on flat
    n_ramps = 6
    pit_depth = -1.0                                 # Flat ground below ramps (pit)
    up_direction = +1

    mid_y = m_to_idx(width) // 2
    centerline = mid_y
    spawn_length_i = m_to_idx(2.0)
    cur_x = spawn_length_i
    total_x = m_to_idx(length)

    # Keep spawn area flat
    height_field[:spawn_length_i, :] = 0
    goals[0] = [spawn_length_i - m_to_idx(0.5), centerline]

    # Make pit everywhere except ramps
    height_field[spawn_length_i:, :] = pit_depth

    y_offset_range = m_to_idx(0.6 - 0.3 * difficulty) # Lateral offset for ramps

    prev_height = 0.0
    x_range = []
    y_ramps = []
    ramp_signs = []

    for i in range(n_ramps):
        # Alternate ramp up & down direction
        sign = up_direction if i % 2 == 0 else -up_direction

        # Slightly randomize the ramp angle and lateral position
        ramp_angle = np.deg2rad(random.uniform(min_angle, max_angle))
        ramp_length_eff = ramp_length + random.uniform(-0.1, 0.1)
        ramp_length_i = m_to_idx(ramp_length_eff)
        y_offset = random.randint(-y_offset_range, y_offset_range)
        ramp_center = centerline + y_offset
        ramp_left = ramp_center - ramp_width_i // 2
        ramp_right = ramp_left + ramp_width_i

        start_x = cur_x
        end_x = min(start_x + ramp_length_i, total_x - m_to_idx(1.0))
        local_x = np.arange(start_x, end_x)
        n_points = len(local_x)

        # Linear slope for ramp
        slope = sign * np.tan(ramp_angle)
        heights = prev_height + slope * np.linspace(0, ramp_length_eff, n_points)

        # Write ramp
        for idx, x in enumerate(local_x):
            height_field[x, ramp_left:ramp_right] = heights[idx]
            # Pit elsewhere (already set)

        # Save center of the ramp for goals (middle along length)
        ramp_center_x = int(round(start_x + n_points // 2))
        goals[i+1] = [ramp_center_x, ramp_center]
        prev_height = heights[-1]
        cur_x = end_x
        x_range.append((start_x, end_x))
        y_ramps.append(ramp_center)
        ramp_signs.append(sign)

        # Flat landing between ramps (shorter at harder difficulty)
        flat_space = 0.48 - 0.23 * difficulty
        flat_i = m_to_idx(flat_space)
        if cur_x + flat_i > total_x - m_to_idx(1.5):    # Avoid running out of room
            break
        height_field[cur_x:cur_x+flat_i, ramp_left:ramp_right] = prev_height
        cur_x += flat_i

    # Final goal: flat platform at the end
    end_pad = m_to_idx(1.1)
    finish_x = min(cur_x, total_x-end_pad)
    height_field[finish_x:finish_x+end_pad, :] = prev_height
    goals[7] = [finish_x + end_pad // 2, centerline]

    # If fewer than 8 goals placed, fill in with end positions
    for j in range(i + 2, 7):
        goals[j] = [finish_x, centerline]

    return height_field, goals