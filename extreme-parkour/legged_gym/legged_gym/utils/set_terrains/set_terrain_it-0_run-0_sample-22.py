import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Wide staircase hurdles: robot must repeatedly step up, walk short on platforms, and step down."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for the course
    hurdle_height_min, hurdle_height_max = 0.07, 0.28   # Low step to nearly hip-high
    hurdle_height = hurdle_height_min + (hurdle_height_max - hurdle_height_min) * difficulty
    platform_length = 0.7 - 0.2 * difficulty           # How long each upstep is walkable
    hurdle_width = 1.2                                # All obstacles at least 1m wide
    step_thickness = 0.07 + 0.04 * difficulty         # Wall thickness
    gap_length = 1.1 - 0.5 * difficulty               # Space between steps (robot walks on ground)
    n_hurdles = 6                                     # Number of steps/platforms

    # Platform sizes (quantized)
    q_hurdle_height = hurdle_height
    q_platform_length = m_to_idx(platform_length)
    q_step_thickness = m_to_idx(step_thickness)
    q_gap_length = m_to_idx(gap_length)
    q_hurdle_width = m_to_idx(hurdle_width)

    mid_y = m_to_idx(width // 2)
    y1 = mid_y - q_hurdle_width // 2
    y2 = mid_y + q_hurdle_width // 2

    # Clear spawning area
    spawn_end = m_to_idx(2)
    height_field[0:spawn_end, :] = 0

    # First goal: just after spawn
    goals[0] = [spawn_end + m_to_idx(0.2), mid_y]

    # Add hurdles
    cur_x = spawn_end
    for h in range(n_hurdles):
        # Wall ("step") - vertical face
        wall_x1 = cur_x
        wall_x2 = cur_x + q_step_thickness
        # Platform on top
        plat_x1 = wall_x2
        plat_x2 = plat_x1 + q_platform_length

        # Raise wall and platform
        height_field[wall_x1:wall_x2, y1:y2] = q_hurdle_height
        height_field[plat_x1:plat_x2, y1:y2] = q_hurdle_height

        # Robot traverses wall and climbs to platform -- so place goal in the center of the flat platform
        goal_x = (plat_x1 + plat_x2) // 2
        goal_y = mid_y
        # Place the goal right on the platform center
        if h < 7:
            goals[h + 1] = [goal_x, goal_y]

        # Next ground section ("descending step" drops straight down)
        next_ground_x1 = plat_x2
        next_ground_x2 = next_ground_x1 + q_gap_length
        height_field[next_ground_x1:next_ground_x2, :] = 0   # Reset to ground

        # Move forward
        cur_x = next_ground_x2

    # Final goal: just after last hurdle
    goals[-1] = [min(cur_x + m_to_idx(0.5), m_to_idx(length)-1), mid_y]

    # Clip everything within the course bounds
    height_field = height_field[:m_to_idx(length), :m_to_idx(width)]

    return height_field, goals