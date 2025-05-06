import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of urban-style step-over rails placed at increasing heights and variable gaps for quadruped high-stepping and precise pacing."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for step-over rails (like low hurdles or pipes found in urban parks)
    # These are narrow, horizontal "rails" for the robot to step over at pace
    rail_count = 6
    rail_min_gap = 0.5 + 0.7 * difficulty     # Minimum gap between rails, meters
    rail_max_gap = 1.2 + 0.5 * difficulty     # Maximum gap between rails, meters
    rail_length = width - 0.5                 # Rail doesn't span complete width, to encourage lateral accuracy
    rail_width = 0.1 + 0.15 * difficulty      # Rails are always wide enough, but increase slightly with difficulty
    rail_height_min = 0.10 + 0.10 * difficulty   # Lowered at easy, raised at hard
    rail_height_max = 0.15 + 0.18 * difficulty

    x_pos = m_to_idx(2.0)   # Spawn area behind the first obstacle
    y_center = m_to_idx(width / 2)

    # Set spawn (flat ground)
    height_field[0:x_pos, :] = 0
    # First goal: before first rail
    goals[0] = [x_pos - m_to_idx(0.5), y_center]

    goal_idx = 1

    # A helper for placing a "rail" obstacle (a narrow line/surface along the y axis at given x)
    def place_rail(x, rail_length, rail_width, height):
        mid_y = m_to_idx(width/2)
        rail_hw = m_to_idx(rail_width / 2)
        rail_hlen = m_to_idx(rail_length / 2)
        y1 = int(mid_y - rail_hlen)
        y2 = int(mid_y + rail_hlen)
        # place the rail along x at given height
        height_field[x:x+m_to_idx(0.12), y1:y2] = height

    # Build the obstacle course: rails and goals
    for i in range(rail_count):
        # Save last rail's x for goal calculation
        last_x_pos = x_pos

        # Place the rail (randomize its height and slight y offset for advancing difficulty)
        current_rail_height = np.random.uniform(rail_height_min, rail_height_max)
        y_offset = random.randint(-m_to_idx(0.2 + 0.2 * difficulty), m_to_idx(0.2 + 0.2 * difficulty))
        rail_top_x = x_pos
        # Offset the rail in y (so robot cannot just blindly strafe, must aim legs)
        rail_mid_y = y_center + y_offset
        rail_hlen = m_to_idx(rail_length / 2)
        y1 = int(rail_mid_y - rail_hlen)
        y2 = int(rail_mid_y + rail_hlen)
        height_field[rail_top_x:rail_top_x+m_to_idx(0.12), y1:y2] = current_rail_height

        # Place a "goal" just after the rail, in its center (encourage robot to step over then pause at safe spot)
        goals[goal_idx] = [rail_top_x + m_to_idx(0.3), rail_mid_y]
        goal_idx += 1
        # Advance to the next rail position (gap increases with difficulty and some randomness)
        gap = np.random.uniform(rail_min_gap, rail_max_gap)
        x_pos = int(rail_top_x + m_to_idx(0.13) + m_to_idx(gap))

    # After last rail, provide "landing zone" and set last goals
    landing_zone_x = x_pos
    # Flat area after obstacles
    height_field[landing_zone_x:, :] = 0

    # Fill remaining goals (up to 8)
    while goal_idx < 8:
        next_goal_x = int(landing_zone_x + m_to_idx(0.4 + 0.5 * (goal_idx-rail_count)))
        if next_goal_x >= height_field.shape[0]:
            # If we run out of space, cap at edge
            next_goal_x = height_field.shape[0] - m_to_idx(1)
        goals[goal_idx] = [next_goal_x, y_center]
        goal_idx += 1

    return height_field, goals