import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Sequence of elevated, narrow balance beams with periodic 90-degree turns for testing precise foot placement and turning control."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Field setup
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters
    # Balance beam parameters scale with difficulty:
    # Higher: narrower and higher; Lower: wider and lower.
    beam_width = 0.7 - 0.3 * difficulty       # from 0.7m (easy) to 0.4m (hard; must be >= 0.4m)
    beam_width = max(0.4, beam_width)
    beam_width_idx = m_to_idx(beam_width)
    beam_height = 0.05 + 0.25 * difficulty    # from 0.05m (easy, curb) to 0.3m (hard, step up)
    beam_length = 2.1 - 1.0 * difficulty      # from 2.1m (easy, long) to 1.1m (hard, short)
    beam_length = max(1.1, beam_length)
    beam_length_idx = m_to_idx(beam_length)

    mid_y = m_to_idx(width) // 2

    # Locations for beams (alternating horizontal/vertical for zig-zag)
    # Always keep beams inside bounds (leave 0.2m clearance from edges)
    side_clearance = m_to_idx(0.2)
    cur_x = m_to_idx(1.0)  # Start after spawn-safe area
    cur_y = mid_y

    # Clear spawn area
    spawn_limit_x = m_to_idx(2.0)
    height_field[:spawn_limit_x, :] = 0

    # Setup initial goal at the starting location
    goals[0] = [m_to_idx(1.0), cur_y]

    direction = 0  # 0: right (+x), 1: up (+y), 2: left (-x), 3: down (-y)
    increments = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    num_beams = 7

    for i in range(num_beams):
        # Set beam center location
        if direction % 2 == 0:
            # Beam along x (horizontal)
            x1 = cur_x
            x2 = min(x1 + beam_length_idx, height_field.shape[0] - side_clearance)
            y1 = max(cur_y - beam_width_idx // 2, side_clearance)
            y2 = min(cur_y + (beam_width_idx + 1) // 2, height_field.shape[1] - side_clearance)
            height_field[x1:x2, y1:y2] = beam_height
            # Set goal for this beam near far end center
            goal_x = x2 - m_to_idx(0.5)
            goal_y = (y1 + y2) // 2
            goals[i+1] = [goal_x, goal_y]
            # Move to far end
            cur_x = x2 - 1  # -1 for overlap
            cur_y = goal_y
        else:
            # Beam along y (vertical)
            y1 = cur_y
            y2 = min(y1 + beam_length_idx, height_field.shape[1] - side_clearance)
            x1 = max(cur_x - beam_width_idx // 2, side_clearance)
            x2 = min(cur_x + (beam_width_idx + 1)//2, height_field.shape[0] - side_clearance)
            height_field[x1:x2, y1:y2] = beam_height
            # Set goal for this beam near far end center
            goal_x = (x1 + x2) // 2
            goal_y = y2 - m_to_idx(0.5)
            goals[i+1] = [goal_x, goal_y]
            # Move to far end
            cur_x = goal_x
            cur_y = y2 - 1

        # Turn 90 degrees (right turn for odd-indexed beam)
        direction = (direction + 1) % 4

    # Final goal: just after last beam, on flat ground
    last_x = min(cur_x + m_to_idx(0.7), height_field.shape[0] - side_clearance)
    last_y = min(cur_y + m_to_idx(0.7), height_field.shape[1] - side_clearance)
    goals[7] = [last_x, last_y]

    # Make terrain surrounding beams (except spawn) a negative pit
    pit_depth = -0.25 - 0.5 * difficulty
    # For all indices > spawn_limit_x, set to pit except where height_field > 0 (the beams)
    pit_mask = (height_field == 0)
    height_field[spawn_limit_x:, :][pit_mask[spawn_limit_x:, :]] = pit_depth

    return height_field, goals