import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """U-shaped urban parkour ledge: Run forward, sharp left on balance beam, sharp right on balance beam, jump off."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course layout: U-Shape using urban 'ledges' and 'balancing beams'
    # 1. Run straight, climb up curb/ledge (height depends on difficulty)
    # 2. Sharp left onto a narrow but traversable balance beam
    # 3. Sharp right onto another beam, then jump off

    curb_height = 0.09 + 0.16 * difficulty     # Simulates a curb to climb onto (9-25cm)
    beam_height = curb_height                  # Beams stay at curb height for alignment
    beam_width = 0.32 - 0.10 * difficulty      # Beam gets narrower (32-22cm)
    beam_length = 2.6 + 2.2 * difficulty       # Longer balance at high difficulty (2.6-4.8m)
    landing_height = 0                         # After final jump down
    ledge_length = 1.2                         # Ledge length before beam (1.2m)
    gap_length = 0.17 + 0.20 * difficulty      # Gap that must be jumped across after beam (17-37cm)

    side_margin = 0.34                        # amount of clearance from y-edge, in meters
    start_clear = 2.0                         # flat clear ground at start for spawning

    # Indexing
    x0 = m_to_idx(0)
    x_spawn_end = m_to_idx(start_clear)
    x_ledge_end = x_spawn_end + m_to_idx(ledge_length)
    y_mid = m_to_idx(width / 2)
    y_margin = m_to_idx(side_margin)
    beam_w = max(m_to_idx(beam_width), m_to_idx(0.24)) # never less than 24cm wide

    y_left = y_margin + beam_w // 2
    y_right = m_to_idx(width) - y_margin - beam_w // 2

    # --------- 1. Flat spawn area ---------
    height_field[x0:x_spawn_end, :] = 0
    goals[0] = [m_to_idx(1.0), y_mid]   # initial goal in flat region for heading alignment

    # --------- 2. Forward ledge climb ---------
    height_field[x_spawn_end:x_ledge_end, y_mid - m_to_idx(0.7):y_mid + m_to_idx(0.7)] = curb_height
    goals[1] = [x_spawn_end + m_to_idx(0.6), y_mid]    # center of the ledge

    # --------- 3. 90-degree left: First balance beam ---------
    x_beam_start = x_ledge_end
    x_beam_end = x_beam_start + m_to_idx(beam_length * 0.37)   # 1st leg of beam (turn left)
    y_beam_left_start = y_mid - beam_w // 2
    y_beam_left_end = y_margin + beam_w

    # Beam goes to left edge in y, robot must make about a 90-deg left
    height_field[x_beam_start:x_beam_end, y_beam_left_end - beam_w : y_beam_left_end] = beam_height
    goals[2] = [x_beam_end - m_to_idx(0.18), y_beam_left_end - beam_w // 2]   # near left edge: turn point

    # --------- 4. 90-degree right: Second beam ---------
    # Now move sideways along y at the left, toward the far wall
    y_beam_right_start = y_beam_left_end - beam_w
    y_beam_right_end = y_beam_right_start + m_to_idx(beam_length)
    x_beam2 = x_beam_end                              # same x
    height_field[x_beam2:x_beam2 + beam_w, 
                 y_beam_right_start:y_beam_right_end] = beam_height
    goals[3] = [x_beam2 + beam_w // 2, y_beam_right_end - m_to_idx(0.2)]   # near end of beam: turn point

    # --------- 5. 90-degree right: Third beam forward ---------
    # Head forward on rightmost side
    x_beam3_start = x_beam2 + beam_w
    x_beam3_end = x_beam3_start + m_to_idx(beam_length * 0.42)
    y_beam3 = y_beam_right_end - beam_w
    height_field[x_beam3_start:x_beam3_end, y_beam3:y_beam3 + beam_w] = beam_height
    goals[4] = [x_beam3_end - m_to_idx(0.18), y_beam_right_end - beam_w // 2]

    # --------- 6. Gap to final landing ---------
    x_gap_start = x_beam3_end
    x_gap_end = x_gap_start + m_to_idx(gap_length)
    # No surface on the gap - robot must jump!
    height_field[x_gap_start:x_gap_end, y_beam3:y_beam3 + beam_w] = -0.5

    # Landing zone after final jump
    x_land_start = x_gap_end
    x_land_end = min(m_to_idx(length), x_land_start + m_to_idx(2.5))
    height_field[x_land_start:x_land_end, y_beam3:y_beam3 + m_to_idx(1.0)] = landing_height
    goals[5] = [x_land_start + m_to_idx(0.45), y_beam_right_end - beam_w // 2] # after the jump

    # --------- 7. Optional small drop or step at end ---------
    if difficulty > 0.4:
        final_drop_x0 = x_land_end
        final_drop_x1 = min(m_to_idx(length), final_drop_x0 + m_to_idx(0.65 + 0.35 * difficulty))
        height_field[final_drop_x0:final_drop_x1, y_beam3:y_beam3 + m_to_idx(1.0)] = -0.22
        final_goal_x = final_drop_x1 - m_to_idx(0.1)
        final_goal_y = y_beam_right_end - beam_w // 2
    else:
        final_goal_x = x_land_end - m_to_idx(0.1)
        final_goal_y = y_beam_right_end - beam_w // 2
    goals[6] = [final_goal_x, final_goal_y]

    # --------- Fill in last goal at the course exit ---------
    goals[7] = [m_to_idx(length) - m_to_idx(0.3), y_beam_right_end - beam_w // 2]

    # --------- Clamp bounds to avoid IndexError ---------
    height_field = height_field[:m_to_idx(length), :m_to_idx(width)]
    for i in range(goals.shape[0]):
        goals[i, 0] = np.clip(goals[i, 0], 0, m_to_idx(length)-1)
        goals[i, 1] = np.clip(goals[i, 1], 0, m_to_idx(width)-1)
    
    return height_field, goals