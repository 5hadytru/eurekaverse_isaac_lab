import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping-stone 'city crossing' with narrow beams, curbs, and angled turns to test precise foot placement and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    #####################
    # Course Parameters #
    #####################

    # Beam parameters (beams as narrow walkways over pits)
    beam_width = 0.45 + 0.3*(1-difficulty)    # Narrows with difficulty, min: 0.45, max: 0.75m
    beam_width_idx = m_to_idx(beam_width)

    beam_height = 0.12 + 0.09*difficulty      # Beam sits above a small pit
    pit_depth = -(0.35 + 0.30*difficulty)     # Pit below = negative field

    # Curb parameters (low, wide steps)
    curb_width = 1.4
    curb_length = 0.7 + 0.5*(1-difficulty)
    curb_width_idx = m_to_idx(curb_width)
    curb_length_idx = m_to_idx(curb_length)
    curb_height = 0.13 + 0.11*difficulty

    mid_y = m_to_idx(width/2)
    total_x = m_to_idx(length)
    total_y = m_to_idx(width)

    # Offset for safe spawning
    spawn_length_idx = m_to_idx(2)
    height_field[:spawn_length_idx, :] = 0
    goals[0] = [m_to_idx(1), mid_y]  # spawn point

    ###############
    # Beams/Pits  #
    ###############
    beam_segments = 2 + int(2 * difficulty)   # 2 beams at easy, up to 4 at hard
    beam_length = (length - 5.0) / (beam_segments+1)  # leave space for curbs at both ends
    beam_length_idx = m_to_idx(beam_length)
    pit_length = 0.48 + 0.55*difficulty
    pit_length_idx = m_to_idx(pit_length)

    curx_idx = spawn_length_idx
    prev_goal = [curx_idx, mid_y]
    goal_num = 1

    for seg in range(beam_segments):
        # Place a pit
        pit_start_x = curx_idx
        pit_end_x = pit_start_x + pit_length_idx
        height_field[pit_start_x:pit_end_x, :] = pit_depth

        # Place beam over pit
        # For challenge, offset the beam side to side at each segment
        if seg % 2 == 0:
            beam_center_y = mid_y - m_to_idx(0.8 + 0.6*difficulty) // 2
        else:
            beam_center_y = mid_y + m_to_idx(0.8 + 0.6*difficulty) // 2

        beam_start_x = pit_start_x
        beam_end_x = pit_end_x
        by1 = beam_center_y - beam_width_idx//2
        by2 = beam_center_y + beam_width_idx//2

        # Ensure within bounds
        by1 = max(by1, m_to_idx(0.1))
        by2 = min(by2, total_y - m_to_idx(0.1))
        height_field[beam_start_x:beam_end_x, by1:by2] = beam_height

        # Place goal at middle of beam
        if goal_num < 8:
            gx = (beam_start_x + beam_end_x)//2
            gy = (by1 + by2)//2
            goals[goal_num] = [gx, gy]
            prev_goal = [gx, gy]
            goal_num += 1

        curx_idx = pit_end_x + m_to_idx(0.18 + 0.1*difficulty)  # Slight gap for flat ground

        # Flat area for stability on landing
        height_field[curx_idx:curx_idx+m_to_idx(0.45), :] = 0

        curx_idx += m_to_idx(0.45)

    ##############
    # Curb Steps #
    ##############
    # Place final curb series requiring single or double step-up, maybe with turns

    curb_count = 2 if difficulty < 0.5 else 3
    turn_angle = 0.45 + 0.5*difficulty  # Amount of "turn" at each curb step
    curb_start_y = mid_y

    for curb in range(curb_count):
        cl = curb_length_idx
        cw = curb_width_idx
        ch = curb_height

        curb_x1 = curx_idx
        curb_x2 = curb_x1 + cl

        # Place curb step at angle: left, right, center
        direction = -1 if curb % 2 == 0 else 1
        offset = direction * m_to_idx(turn_angle * (curb+1) * (1-difficulty+0.5))
        curb_y1 = curb_start_y + offset - cw//2
        curb_y2 = curb_start_y + offset + cw//2
        curb_y1 = max(m_to_idx(0.1), min(total_y - cw - m_to_idx(0.1), curb_y1))
        curb_y2 = curb_y1 + cw

        # Raise the platform for curb
        height_field[curb_x1:curb_x2, curb_y1:curb_y2] = ch

        # Place goal at center of curb step
        if goal_num < 8:
            gx = (curb_x1 + curb_x2)//2
            gy = (curb_y1 + curb_y2)//2
            goals[goal_num] = [gx, gy]
            prev_goal = [gx, gy]
            goal_num += 1

        # Move curx_idx for next curb
        curx_idx = curb_x2

    # Final goal after last curb
    if goal_num < 8:
        final_goal_x = min(curx_idx + m_to_idx(0.9), height_field.shape[0]-1)
        final_goal_y = prev_goal[1]
        goals[goal_num] = [final_goal_x, final_goal_y]

    # Fill unused goals with last goal position (so the array is always size 8)
    for i in range(goal_num+1, 8):
        goals[i] = goals[goal_num]

    # Ensure entire area is padded below 0 if necessary
    # Make sure spawn and finish zones are always flat
    height_field[0:spawn_length_idx, :] = 0
    height_field[-m_to_idx(1):, :] = 0

    return height_field, goals