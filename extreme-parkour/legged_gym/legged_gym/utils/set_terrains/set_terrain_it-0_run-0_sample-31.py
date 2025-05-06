import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of balance beams and zig-zag turns testing lateral stability and precise turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for the balance beam course
    # Beams are narrow, so width 0.4-0.5m, length 2m. At higher difficulty, narrower and higher off ground.
    min_beam_width = 0.5 - 0.25 * difficulty     # 0.5m (easy) -> 0.25m (hard)
    beam_width = max(min_beam_width, 0.24)       # ensure no less than 0.24m (minimum allowed, a bit wider than robot body)
    beam_length = 2.0 - 0.25 * difficulty        # 2.0m -> 1.75m, shorter at high difficulty for more direction changes
    beam_height = 0.10 + 0.20 * difficulty       # 0.10m (easy) -> 0.30m (hard, can't just walk off/on)
    gap_between_beams = 0.10 + 0.25 * difficulty  # 0.1m (easy) -> 0.35m (hard)
    spawn_length = m_to_idx(2.0)
    mid_y = m_to_idx(width/2)

    n_beams = 4  # Zig-zag: 4 straight beams, robot zig-zags left/right
    beam_dirs = [0, 1, 0, -1]  # alternate straight, right, straight, left
 
    # Place spawn area
    height_field[:spawn_length, :] = 0
    goals[0] = [spawn_length//2, mid_y]  # first goal is at the spawn

    cur_x = spawn_length
    cur_y = mid_y
    beam_w = m_to_idx(beam_width)
    beam_l = m_to_idx(beam_length)
    beam_h = beam_height
    gap = m_to_idx(gap_between_beams)
    zig_offset = m_to_idx( (1.1 - 0.4*difficulty) )  # how far sideways to offset at the zig/zag (easy: 1.1m, hard: 0.7m)

    for i in range(n_beams):
        # Zig or zag
        if beam_dirs[i] == 0:
            # Straight
            next_y = cur_y
        else:
            next_y = cur_y + beam_dirs[i]*zig_offset
            # Clamp to within field
            next_y = np.clip(next_y, m_to_idx(beam_width)//2, m_to_idx(width) - m_to_idx(beam_width)//2 - 1)

        # The beam runs from cur_x to cur_x+beam_l, at current y, with width beam_w
        y1 = int(np.clip(cur_y - beam_w//2, 0, m_to_idx(width)-1))
        y2 = int(np.clip(cur_y + beam_w//2, 0, m_to_idx(width)-1))
        x1 = int(cur_x)
        x2 = int(np.clip(cur_x + beam_l, 0, m_to_idx(length)-1))
        height_field[x1:x2, y1:y2] = beam_h

        # Put goal in center of this beam
        goals[i+1] = [ (x1 + x2)//2, (y1 + y2)//2 ]

        # Update position: "hop" gap ahead; set next beam's center to new y
        cur_x = x2 + gap
        cur_y = next_y

        # Add a "pit" between beams so robot cannot drop down and re-climb
        pit_depth = -0.8 - 0.5*difficulty  # deep pit to force balancing
        pit_x1 = x2
        pit_x2 = int(np.clip(cur_x, 0, m_to_idx(length)-1))
        height_field[pit_x1:pit_x2, :] = pit_depth

    # Add a final wide (1.5m) landing platform at the end
    platform_length = m_to_idx(1.2)
    platform_width = m_to_idx(1.5)
    x1 = int(cur_x)
    x2 = int(np.clip(cur_x + platform_length, 0, m_to_idx(length)))
    y1 = int(np.clip(cur_y - platform_width//2, 0, m_to_idx(width)-1))
    y2 = int(np.clip(cur_y + platform_width//2, 0, m_to_idx(width)-1))
    height_field[x1:x2, y1:y2] = 0
    goals[5] = [ (x1 + x2)//2, (y1 + y2)//2 ]

    # The rest of the goals orient the finish--finish is simply at end platform far edge
    for g in range(6, 8):
        # Spread last two goals out along the wide end platform for straight finish
        gx = int( x2 - (g-5)*m_to_idx(0.2) )
        gy = (y1+y2)//2
        goals[g] = [gx, gy]

    # If needed, pad in the final goal in case not filled
    if (n_beams+1) < 8:
        for g in range(n_beams+1, 8):
            goals[g] = goals[n_beams]

    return height_field, goals