import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of alternating (offset) wide low balance beams and narrow planks for precise foot placement—tests quadruped's dynamic walking and lateral stabilization."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course Parameters
    spawn_length = m_to_idx(2.0)  # Spawn area length (no obstacles)
    n_obstacles = 6  # Six balance features between goals
    course_length_idx = m_to_idx(length)
    course_width_idx = m_to_idx(width)
    mid_y = course_width_idx // 2

    # Obstacle types: wide beams (stable but require turning and precision), and narrow (precision planks)
    # Wide beams—require balancing and some slight turning between them
    beam_length_m = 1.8 - 0.6 * difficulty        # length along x
    beam_width_m = 1.2 - 0.6 * difficulty         # width along y (gets thinner)
    beam_height_m = 0.08 + 0.07 * difficulty      # elevated above ground

    # Plank (narrow path) parameters—requires very careful balance
    plank_length_m = 1.3 + 0.3 * difficulty
    plank_width_m = 0.4 + 0.1 * (1 - difficulty)  # always at least 0.4m
    plank_height_m = beam_height_m
    gap_length_m = 0.20 + 0.45 * difficulty       # gaps between obstacle features

    # Ground outside beams/planks is lowered to -0.9—forcing the quadruped to stay on track!
    height_field[spawn_length:, :] = -0.9

    # Initialize x-position
    cur_x = spawn_length
    goal_idx = 0

    # Place initial goal at the end of spawn zone, in the middle
    goals[goal_idx] = [cur_x, mid_y]
    goal_idx += 1

    # Alternate beam and plank; also alternate beam offset to left/right for turning
    for obs in range(n_obstacles):
        if obs % 2 == 0:  # Wide balance beam
            l = m_to_idx(beam_length_m)
            w = m_to_idx(beam_width_m)
            h = beam_height_m

            x1, x2 = cur_x, cur_x + l
            # Alternate side offset for balance beams
            y_offset = m_to_idx(0.5) if (obs//2) % 2 == 0 else -m_to_idx(0.5)
            beam_cy = mid_y + y_offset
            y1 = max(0, beam_cy - w//2)
            y2 = min(course_width_idx, beam_cy + w//2)

            height_field[x1:x2, y1:y2] = h
            
            # Set goal at the end-center of the beam
            goals[goal_idx] = [x2 - m_to_idx(0.25), beam_cy]
            goal_idx += 1
            cur_x += l

        else:  # Narrow plank
            l = m_to_idx(plank_length_m)
            w = m_to_idx(plank_width_m)
            h = plank_height_m

            x1, x2 = cur_x, cur_x + l
            # Place plank at center, but scatter a little up or down to add variety
            side_nudge = m_to_idx(0.25) * (-1 if random.random() > 0.5 else 1)
            plank_cy = mid_y + side_nudge
            y1 = max(0, plank_cy - w//2)
            y2 = min(course_width_idx, plank_cy + w//2)

            height_field[x1:x2, y1:y2] = h
            # Set goal at end-center of the plank
            goals[goal_idx] = [x2 - m_to_idx(0.15), plank_cy]
            goal_idx += 1
            cur_x += l

        # Add a gap (pit) after every feature (except after the last one)
        if obs < n_obstacles - 1:
            cur_x += m_to_idx(gap_length_m)

    # Make sure last goal is at a safe exit spot (flat ground)
    flat_exit_len = m_to_idx(1.0)
    height_field[cur_x:cur_x+flat_exit_len, :] = 0
    goals[7] = [min(course_length_idx-2, cur_x + flat_exit_len//2), mid_y]

    # If not all 8 goals filled (can happen at low difficulty), duplicate last
    for k in range(goal_idx, 8):
        goals[k] = goals[goal_idx-1]

    return height_field, goals