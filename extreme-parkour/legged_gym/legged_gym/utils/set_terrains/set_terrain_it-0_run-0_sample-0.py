import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Alternating ramps and low rails for testing balance and grip under angled ground contact."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Ramps/rail parameters
    # Ramps: angled surfaces requiring step or bound up/down (increase steepness with difficulty)
    max_angle_deg = 25 + 20 * difficulty  # up to 45deg at hardest
    max_angle_rad = np.deg2rad(max_angle_deg)
    ramp_height = np.tan(max_angle_rad) * 1.5     # ramp runs 1.5m in length
    ramp_length = 1.5
    ramp_width = 1.4 + 0.7 * (1-difficulty)       # slightly wider at easy settings

    # Rails: raised narrow low-profile beams, force careful foot placement for balance
    rail_width = 0.3 + 0.2 * (1-difficulty)       # easier: up to 0.5m wide, hard: 0.3m
    rail_height = 0.12 + 0.10 * difficulty
    rail_length = 1.1
    rail_gap = 0.28 + 0.24 * difficulty           # pit to punish LOSING balance
    
    mid_y = m_to_idx(width / 2)

    # Spawn zone: keep flat
    spawn_x = m_to_idx(2)
    height_field[:spawn_x, :] = 0
    goals[0] = [m_to_idx(1), mid_y]

    # Starting positions for obstacles
    cur_x = spawn_x

    for i in range(1, 7, 2):  # We will place 3 ramp-rail pairs, mapping to 6 goals plus spawn and exit
        ########################
        # Add a ramp
        ########################
        ramp_x1 = cur_x
        ramp_x2 = ramp_x1 + m_to_idx(ramp_length)

        # Ramp vertical displacement can be up (if even) and then down (if odd), to test both ascent and descent
        if (i//2) % 2 == 0:
            ramp_dir = 1  # up
        else:
            ramp_dir = -1 # down

        # Linear slope along ramp axis
        ramp_y1 = mid_y - m_to_idx(ramp_width/2)
        ramp_y2 = mid_y + m_to_idx(ramp_width/2)
        for rx in range(ramp_x1, ramp_x2):
            rel = (rx - ramp_x1) / max(1, ramp_x2 - ramp_x1-1)
            height_field[rx, ramp_y1:ramp_y2] = ramp_dir * ramp_height * rel + height_field[ramp_x1, ramp_y1]

        # Goals at top of each ramp
        goals[i] = [ (ramp_x1 + ramp_x2)//2, mid_y ]

        cur_x = ramp_x2

        ########################
        # Add a rail
        ########################
        rail_x1 = cur_x
        rail_x2 = rail_x1 + m_to_idx(rail_length)
        rail_y1 = mid_y - m_to_idx(rail_width/2)
        rail_y2 = mid_y + m_to_idx(rail_width/2)
        # Rail is a raised beam with a pit on either side
        height_field[rail_x1:rail_x2, :] = -rail_gap  # fill with pit (negative height)
        height_field[rail_x1:rail_x2, rail_y1:rail_y2] = height_field[ramp_x2-1, rail_y1] + rail_height

        # Place goal at center of rail
        goals[i+1] = [ (rail_x1 + rail_x2)//2, mid_y ]

        cur_x = rail_x2

    # Final flat area for finish
    end_x = m_to_idx(length)
    if cur_x < end_x:
        height_field[cur_x:end_x, :] = 0
        goals[7] = [cur_x + m_to_idx(0.9), mid_y]
    else:
        goals[7] = [end_x-1, mid_y]

    return height_field, goals