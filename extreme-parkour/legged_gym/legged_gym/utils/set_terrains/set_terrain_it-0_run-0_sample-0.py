import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A zig-zag balance beam course over pits, testing precise foot placement and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course Parameters
    beam_width = 0.22 + 0.18 * (1 - difficulty)   # meters, 0.4m at easy, 0.22m at hard
    beam_width_idx = m_to_idx(beam_width)
    beam_length = 2.4 - 0.8 * difficulty          # meters, longer on easy
    beam_length_idx = m_to_idx(beam_length)
    # Vertical offset for zig-zag
    beam_offset = m_to_idx(0.9 + 0.9 * difficulty)  # How far up/down zig-zags, wider at hard
    gap = 0.3 + 0.6 * difficulty                  # meters, gaps between beams get bigger with difficulty
    gap_idx = m_to_idx(gap)

    mid_y = m_to_idx(width/2)
    x_start = m_to_idx(2)      # Do not place obstacle before spawn
    spawn_idx = m_to_idx(2)
    height_field[:spawn_idx, :] = 0  # Flat spawn

    beam_height = 0.08 + 0.13 * difficulty    # beams rise with difficulty

    # Set all non-beam regions (after spawn) to be significant pits
    height_field[spawn_idx:, :] = -0.7 - difficulty * 0.5

    # Zig-zag beam centers (alternating left/right)
    y_centers = []
    y0 = mid_y
    for i in range(4):
        if i % 2 == 0:
            y_centers.append(y0 + beam_offset)
        else:
            y_centers.append(y0 - beam_offset)

    # Each beam is one stage; we will fit 8 turns (7 bends).
    # Each leg has a beam with a turn in between.
    x = x_start
    goal_idx = 0
    # Place initial goal at spawn
    goals[goal_idx] = [m_to_idx(1), mid_y]
    goal_idx += 1

    for i in range(4):
        # Draw beam
        x_end = min(m_to_idx(length) - 1, x + beam_length_idx)
        y_c = y_centers[i]
        half_bw = beam_width_idx // 2
        y1 = max(0, y_c - half_bw)
        y2 = min(m_to_idx(width), y_c + half_bw)

        height_field[x:x_end, y1:y2] = beam_height

        # Middle of beam is the goal
        g_x = x + (x_end-x)//2
        g_y = y_c
        goals[goal_idx] = [g_x, g_y]
        goal_idx += 1

        # Bend/turn gap -- short "bridge"/platform at a corner (0.5m)
        if i < 3:
            turn_x1 = x_end
            turn_x2 = min(turn_x1 + m_to_idx(0.5), m_to_idx(length)-1)
            y1 = min(y_centers[i], y_centers[i+1]) - half_bw
            y2 = max(y_centers[i], y_centers[i+1]) + half_bw
            y1 = max(0, y1)
            y2 = min(m_to_idx(width), y2)
            height_field[turn_x1:turn_x2, y1:y2] = beam_height

            # Add a goal at the center of the 'bend'
            g_x = (turn_x1 + turn_x2) // 2
            g_y = (y_centers[i] + y_centers[i+1]) // 2
            goals[goal_idx] = [g_x, g_y]
            goal_idx += 1

            # Next beam should start after the turn
            x = turn_x2 + gap_idx
        else:
            # If last beam, x just advances to end
            x = x_end

    # Final goal: after last beam, back to middle at the end of the course
    final_goal_x = min(m_to_idx(length)-2, x + m_to_idx(0.7))
    goals[7] = [final_goal_x, mid_y]
    # Give final flat exit
    height_field[x:, :] = 0

    return height_field, goals