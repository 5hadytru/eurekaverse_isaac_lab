import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Zig-zag ramp gauntlet: Multiple wide ramps at sharp angles requiring climbing, descending, and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))
    # Ramp properties scale with difficulty
    total_length_idx = m_to_idx(length)
    total_width_idx = m_to_idx(width)
    spawn_x_idx = m_to_idx(1)
    mid_y_idx = total_width_idx//2
    ramp_length_m = 2.2 - 0.5 * difficulty   # ramp too steep if too short!
    ramp_width_m  = 1.2 + 0.8 * difficulty   # wider at high diff: more options, less side-stepping at low diff
    ramp_height   = 0.12 + 0.23 * difficulty # higher at higher difficulty: 12cm to 35cm

    ramp_length = m_to_idx(ramp_length_m)
    ramp_width  = m_to_idx(ramp_width_m)

    turn_offset = m_to_idx(1.0) # how far 'vertically' the ramp shifts per segment

    # set spawn area clear
    height_field[:spawn_x_idx+1,:] = 0

    # Place first goal: start, unshifted
    cur_x = spawn_x_idx
    cur_y = mid_y_idx
    goals[0] = [cur_x, cur_y]

    directions = [+1, -1] * 4  # left, right, left, right, ... (up to 8 segments)

    # Main zig-zag ramp loop
    for i in range(7):
        # Compute bounding box for the ramp
        x0 = cur_x
        y0 = cur_y
        x1 = min(cur_x + ramp_length, total_length_idx - 1)
        # Side offset (zig/zag)
        dir = directions[i]
        y1 = np.clip(cur_y + int(dir * turn_offset), m_to_idx(0.5), total_width_idx-m_to_idx(0.5))

        # The ramp is a rectangle between y0 and y1, interpolating in y as it advances in x
        ramp_min_y = int(min(y0, y1) - ramp_width//2)
        ramp_max_y = int(max(y0, y1) + ramp_width//2)
        ramp_min_y = max(ramp_min_y, 0)
        ramp_max_y = min(ramp_max_y, total_width_idx-1)
        
        # Draw the ramp as a sloped plane (linear in x) from (cur_x, y0) to (x1, y1)
        # Each position (x, y) on the ramp gets an interpolated y center and a corresponding height
        for xi in range(x0, x1):
            frac = (xi - x0) / max(x1 - x0, 1)
            y_center = int(round((1-frac)*y0 + frac*y1))
            ramp_y_start = max(y_center - ramp_width//2, 0)
            ramp_y_end   = min(y_center + ramp_width//2, total_width_idx-1)
            ramp_h = ramp_height * frac # start of ramp is 0m, end is full height

            # Make the ramp rise then flatten then descend in next section
            if i % 2 == 0:
                # Ascend
                height_field[xi, ramp_y_start:ramp_y_end] = ramp_h
            else:
                # Descend
                height_field[xi, ramp_y_start:ramp_y_end] = ramp_height - ramp_h

        # Update next goal: at end of ramp in center
        next_x = x1
        next_y = int(round(y1))
        goals[i+1] = [next_x, next_y]
        cur_x, cur_y = next_x, next_y

    # Final flat goal at the end, set as last goal
    end_x = min(cur_x+m_to_idx(1), total_length_idx-1)
    goals[-1] = [end_x, cur_y]
    # Flat area at finish
    height_field[end_x:, :] = 0

    return height_field, goals