import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Narrow 'balance beams' span pits, requiring precise walking and controlled turning between parallel beams."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for beams and pits (all measurements in meters first)
    pit_depth = -1.1  # Always deep enough to prevent escape
    spawn_clear_length = 2.0   # Spawn is always flat
    beam_length = 1.5 - 0.5 * difficulty # Each beam up to 1.5m, shorter as diff increases
    beam_width = 0.45 if difficulty < 0.4 else (0.40 + 0.3 * (1-difficulty))  # Min width = 0.4m, slightly wider for easy
    beam_height = 0.08 + 0.12 * difficulty       # beams are raised more at high diff
    pit_gap = 0.4 + 0.5 * difficulty             # Size of pit between beams, grows w/ difficulty
    beam_clear_margin = 0.2  # leave 0.2m from edges for safety

    n_beams = 4  # 4 beams w/ lateral offset
    course_length = spawn_clear_length + n_beams * (beam_length + pit_gap) + 1.0  # Ensure fits in 12m

    length_idx = m_to_idx(length)
    width_idx = m_to_idx(width)
    spawn_clear_idx = m_to_idx(spawn_clear_length)
    beam_length_idx = m_to_idx(beam_length)
    beam_width_idx = m_to_idx(beam_width)
    pit_gap_idx = m_to_idx(pit_gap)
    beam_clear_margin_idx = m_to_idx(beam_clear_margin)
    
    # Set the spawn area to flat ground
    height_field[0:spawn_clear_idx, :] = 0

    # Beams' centerline lateral locations: force a zig-zag! 
    y_offset_options = np.linspace(0.7, width - 0.7, n_beams+2)[1:-1]   # don't use true edges
    y_offsets = []
    for i in range(n_beams):
        # Alternate sides (zig-zag) w/ random jitter for variety
        y = y_offset_options[i] + random.uniform(-0.12, 0.12)
        y_offsets.append(m_to_idx(y))

    # Mark all areas except spawn as deep pits initially
    height_field[spawn_clear_idx:, :] = pit_depth

    beam_start_x = spawn_clear_idx
    goals[0] = [m_to_idx(1), m_to_idx(width / 2)]  # Initial goal: spawn

    goal_counter = 1
    for i in range(n_beams):
        # Beam start and end (x)
        x1 = beam_start_x + beam_clear_margin_idx
        x2 = x1 + beam_length_idx
        y_mid = y_offsets[i]

        y1 = max(beam_clear_margin_idx, y_mid - beam_width_idx // 2)
        y2 = min(width_idx - beam_clear_margin_idx, y_mid + beam_width_idx // 2)
        # Place beam: flatten its region and raise to beam_height
        height_field[x1:x2, y1:y2] = beam_height
        
        # Place a goal at the center of each beam
        goals[goal_counter] = [ (x1 + x2)//2, (y1 + y2)//2 ]
        goal_counter += 1
        
        # After each beam (except last), next beam is in a different lateral position with a sharp goal at the end
        if i < n_beams - 1:
            next_y = y_offsets[i+1]
            # Place a goal just after beam, at pit's start, so the agent has to steer
            pit_turn_x = x2 + pit_gap_idx//2
            goals[goal_counter] = [ pit_turn_x, (y_mid + next_y)//2 ]
            goal_counter += 1

        # Move forward for next beam (across pit gap)
        beam_start_x = x2 + pit_gap_idx

    # After last beam, mark remainder as flat and put final goal at finish:
    end_clear_length = m_to_idx(0.7)  # clear area after last beam
    end_start_x = min(beam_start_x, length_idx - end_clear_length)
    height_field[end_start_x:, :] = 0
    # Final goal, at clear finish area, centered
    goals[goal_counter] = [ end_start_x + end_clear_length//2, m_to_idx(width/2) ]

    # If not all 8 goals are filled, keep last goal at finish location
    for k in range(goal_counter+1, 8):
        goals[k] = goals[goal_counter]

    return height_field, goals