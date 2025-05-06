import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of alternating balance beams and narrow planks above a sunken floor, encouraging precise foot placement and straight/turning walking."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]
    
    # Field size and spawn safe zone
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))
    spawn_length = m_to_idx(2)                  # Spawn at 0-2 meters is always flat
    mid_y = m_to_idx(width / 2)                 # Center line
    
    # Parameters for beams/planks (values expand w/difficulty)
    beam_width = 0.30 + 0.10 * (1 - difficulty)   # 0.4m wide at easy, 0.3m at hard, matches rule #5 rare exception
    beam_length = 1.6 + 0.4 * difficulty         # 1.6m at easy, 2m at hard
    plank_width = 0.55 - 0.12 * difficulty       # 0.55m at easy, 0.43m at hard (wider than the robot but narrower than a beam)
    plank_length = 1.0 + 0.7 * difficulty        # Plank 1m-1.7m long
    beam_height = 0.12 + 0.12 * difficulty       # 0.12m high at easy, up to 0.24m at hard
    pit_depth = 0.35 + 0.60 * difficulty         # Sunken floor, deep at hard
    
    gap = 0.20 + 0.10 * difficulty               # Gaps between obstacles, never less than 20cm
    
    # Alternate "beam" (long/narrow) and "plank" (shorter/wider) along path:
    elements = [("beam", beam_length, beam_width, beam_height),    # 1
                ("plank", plank_length, plank_width, beam_height), # 2
                ("beam", beam_length, beam_width, beam_height),    # 3
                ("plank", plank_length, plank_width, beam_height), # 4
                ("beam", beam_length, beam_width, beam_height),    # 5
                ("plank", plank_length, plank_width, beam_height)] # 6

    # Make the floor under beams/planks a pit, enforce flat start/end zones for goal approach
    height_field[spawn_length:, :] = -pit_depth
    height_field[:spawn_length, :] = 0

    cur_x = m_to_idx(2.05)      # Slight buffer after spawn
    obs_count = 0
    goal_idx = 0

    def place_obstacle(x_center, y_center, ext_x, ext_y, height):
        # ext_x, ext_y are half-length/half-width in meters, height is value to write
        x_c = int(round(x_center))
        y_c = int(round(y_center))
        x1 = max(0, x_c - m_to_idx(ext_x))
        x2 = min(height_field.shape[0], x_c + m_to_idx(ext_x))
        y1 = max(0, y_c - m_to_idx(ext_y))
        y2 = min(height_field.shape[1], y_c + m_to_idx(ext_y))
        height_field[x1:x2, y1:y2] = height

    y_path = mid_y  # Start down the center

    while obs_count < len(elements):
        shape, L, W, H = elements[obs_count]

        # Random lateral shift at hard difficulty
        lateral_span = 0.7 * (1 - difficulty)  # Max deviate less at hard
        if shape == "beam" and obs_count % 2 == 1:
            y_shift = random.randint(-m_to_idx(lateral_span), m_to_idx(lateral_span))
            # Turn the path by shifting y_path for planks, so path isn't always straight
            y_path = np.clip(y_path + y_shift, m_to_idx(0.8), m_to_idx(width-0.8))
        else:
            y_shift = 0

        # Lay down the obstacle centered at cur_x, y_path
        place_obstacle(cur_x, y_path, L/2, W/2, H)

        # Set a goal in the center (to line up entry/exit)
        goals[goal_idx] = [cur_x, y_path]
        goal_idx += 1

        # If obstacle is a plank (not a straight-beam), place an extra goal at its end to make the robot turn on/off
        if shape == "plank" and goal_idx < 8:
            end_x = cur_x + (L/2 - 0.2)
            goals[goal_idx] = [end_x, y_path]
            goal_idx += 1

        # Advance to next obstacle: skip over beam/plank plus gap, slightly longer if difficult
        skip_dist = L + gap + random.uniform(0, 0.1)
        cur_x += m_to_idx(skip_dist)
        obs_count += 1

        # Allow space for all obstacles & goals in 12m field
        if cur_x > m_to_idx(length - 1.8):
            break

    # The last goal: before the 12m wall, at the same y_path, on flat ground
    flat_zone_len = m_to_idx(0.7)
    end_x = height_field.shape[0] - flat_zone_len // 2
    height_field[end_x:, :] = 0
    if goal_idx < 8:
        goals[goal_idx] = [end_x, y_path]
        goal_idx += 1

    # Make sure any remaining goals (if less than 8) are placed at the finish line (for API compliance)
    for i in range(goal_idx, 8):
        goals[i] = [end_x, y_path]

    return height_field, goals