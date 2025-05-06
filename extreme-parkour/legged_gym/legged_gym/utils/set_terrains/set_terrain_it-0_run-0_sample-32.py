import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of seesaw (teeter-totter) bridges: tests balancing on unstable, slanted, narrow surfaces."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for the seesaw bridges
    seesaw_length = 1.7 - 0.3 * difficulty  # Length in meters
    seesaw_width = 1.1 - 0.3 * difficulty   # Narrows at high difficulty (min 0.5 m)
    seesaw_width = max(seesaw_width, 0.5)
    seesaw_height = 0.11 + 0.14 * difficulty  # Maximum seesaw height diff from pivot point
    gap_length = 0.35 + 0.6 * difficulty     # Gap between seesaws, these are flat ground
    n_seesaws = 5  # Fewer long seesaws rather than many
    mid_y = m_to_idx(width / 2)

    seesaw_length_idx = m_to_idx(seesaw_length)
    seesaw_width_idx = m_to_idx(seesaw_width)
    gap_length_idx = m_to_idx(gap_length)

    # Set flat safe spawn area 
    spawn_length_idx = m_to_idx(2)
    height_field[:spawn_length_idx, :] = 0
    goals[0] = [spawn_length_idx-m_to_idx(0.5), mid_y]  # Start goal

    cur_x = spawn_length_idx
    seesaw_count = 0

    def add_seesaw(x_start, seesaw_len, y_center, seesaw_wid, tilt_sign):
        """Add a sloped seesaw bridge with the given parameters."""
        x_end = x_start + seesaw_len
        y1 = y_center - seesaw_wid//2
        y2 = y_center + seesaw_wid//2
        # Seesaw pivots at midpoint, so first half ramps up, second half ramps down
        for xi in range(x_start, x_end):
            rel = (xi - x_start) / (seesaw_len - 1)
            if rel < 0.5:
                offs = tilt_sign * 2 * seesaw_height * (rel)
            else:
                offs = tilt_sign * 2 * seesaw_height * (1 - rel)
            height_field[xi, y1:y2] = offs
        return (x_start+x_end)//2, (y1+y2)//2

    # Lay out a chain of seesaws along the center y
    for i in range(n_seesaws):
        # Introduce small random y deviation for some lateral challenge
        y_shift = m_to_idx(random.uniform(-0.35, 0.35) * (1-difficulty))
        # Alternate slant direction
        tilt_sign = 1 if i % 2 == 0 else -1
        seesaw_cx, seesaw_cy = add_seesaw(cur_x, seesaw_length_idx, mid_y + y_shift, seesaw_width_idx, tilt_sign)
        goals[i+1] = [seesaw_cx, seesaw_cy]

        cur_x += seesaw_length_idx
        # Insert gap after bridge
        if i < n_seesaws-1:
            height_field[cur_x:cur_x+gap_length_idx, :] = 0
            cur_x += gap_length_idx

    # Fill any remaining space with flat ground
    height_field[cur_x:, :] = 0
    # Set final (8th) goal after all seesaws
    goals[-1] = [min(cur_x + m_to_idx(0.5), height_field.shape[0]-1), mid_y]

    return height_field, goals