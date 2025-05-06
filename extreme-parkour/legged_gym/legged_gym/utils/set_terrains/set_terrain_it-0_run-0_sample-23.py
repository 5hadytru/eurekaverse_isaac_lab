import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of staggered low rails (beams) above a pit for the robot to balance and traverse."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # ------------------ Course Design -------------------
    # This course tests the quadruped's balance and precise foot placement.
    # The terrain is a pit (height -1.0) crossed by a sequence of narrow (0.4-0.5m wide) rails/beams.
    # Rails alternate in lateral position with modest gaps, providing a zig-zag path forward.
    # The rail width and gap distance become harder with increasing difficulty.

    rail_width = 0.5 - difficulty * 0.1           # width: from 0.5m (easy) to 0.4m (hard)
    rail_height = 0.12 + 0.10 * difficulty        # 0.12-0.22m above ground
    rail_length = 1.5 + 0.5 * difficulty          # Rails get longer with higher difficulty (1.5-2m)
    gap_length = 0.3 + 0.6 * difficulty           # Gaps between rails (0.3-0.9m)
    num_rails = 6                                 # Number of rails to cross

    width_idxs = m_to_idx(width)
    mid_y = width_idxs // 2

    rail_width_idx = m_to_idx(rail_width)
    rail_length_idx = m_to_idx(rail_length)
    gap_length_idx = m_to_idx(gap_length)

    # --- Reset terrain to pit after robot spawn area ---
    spawn_x = m_to_idx(2.0)
    height_field[spawn_x:, :] = -1.0    # Big pit except under rails

    # --- Make spawn area flat (safe start for robot, no obstacles) ---
    height_field[:spawn_x, :] = 0.0
    goals[0] = [m_to_idx(1.5), mid_y]
    
    # Compute possible y positions for zig-zagging rails, left and right from midline
    lateral_shift = int( m_to_idx(0.6 + 0.8 * difficulty) )  # rails can be shifted ±0.6~1.4m from midline
    base_ys = [
        mid_y,
        mid_y + lateral_shift,
        mid_y - lateral_shift,
        mid_y,
        mid_y + lateral_shift//2,
        mid_y - lateral_shift//2,
    ]

    x = spawn_x
    for i in range(num_rails):
        # Determine y-center for this rail (zig-zags)
        y_center = np.clip(base_ys[i % len(base_ys)], rail_width_idx//2, width_idxs-rail_width_idx//2-1)

        rail_x1 = x
        rail_x2 = min(x + rail_length_idx, m_to_idx(length))   # Don't go out of bounds
        rail_y1 = int(y_center - rail_width_idx//2) 
        rail_y2 = int(y_center + np.ceil(rail_width_idx/2)) 

        # Paint rail (elevated path over pit) into the terrain
        height_field[rail_x1:rail_x2, rail_y1:rail_y2] = rail_height

        # Next goal: centered in the middle of this rail
        goals[i+1] = [ (rail_x1 + rail_x2)//2, y_center ]

        # Advance to the next rail start
        x = rail_x2 + gap_length_idx

    # --- Last segment: build a "landing platform" at the end of the course and final goal ---
    # Give the robot a meter-wide, full-width area at ground level
    plat_x1 = min(x, height_field.shape[0] - 1)
    plat_x2 = min(plat_x1 + m_to_idx(1.0), height_field.shape[0] )
    height_field[plat_x1:plat_x2, :] = 0.0
    goals[-1] = [ (plat_x1 + plat_x2)//2, mid_y ]

    # --- Ensure all goal indices are valid integers within bounds ---
    goals = np.clip(goals, [0,0], [height_field.shape[0]-1, height_field.shape[1]-1])

    return height_field, goals