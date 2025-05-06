import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of alternating low hurdles and narrow balance beams testing stepping and stability."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course parameters
    # The quadruped is ~0.645m long, 0.28m wide. Obstacles are sized with respect to this.
    mid_y = m_to_idx(width) // 2

    # Obstacle pattern: hurdle -> beam -> hurdle -> beam ... etc.
    num_obstacles = 6
    safe_margin_x = m_to_idx(2.0)  # Spawn area
    course_x_max = m_to_idx(length)
    course_y_max = m_to_idx(width)

    # Hurdle parameters
    hurdle_width = m_to_idx(1.2)    # Always at least 1m wide
    # Hurdle height increases with difficulty
    hurdle_height = 0.06 + 0.16 * difficulty # 6cm at easy, up to 22cm at hard
    hurdle_length = m_to_idx(0.25 + 0.1 * difficulty)  # Slightly thicker at high diff

    # Beam parameters (balance beam)
    beam_length = m_to_idx(1.1 + 0.4 * difficulty)      # Beams get longer as it gets harder
    # Beam width is challenging but not impossible (0.3 at easy, 0.18 at hard)
    beam_width = m_to_idx(0.3 - 0.12 * difficulty)
    beam_height = 0.10 + 0.10 * difficulty              # 10cm at easy, up to 20cm at hard

    # Distances between obstacles
    gap_between = m_to_idx(0.6 + 0.5 * (1-difficulty))  # Smaller gaps at harder difficulties

    # (x, y) placement tracking
    x = safe_margin_x
    last_y = mid_y

    # Set spawn region flat and goal in center
    height_field[:safe_margin_x, :] = 0
    goals[0] = [safe_margin_x // 2, last_y]

    obstacle_indices = []

    for obs in range(num_obstacles):
        # Alternate between hurdle and beam, place a goal after each obstacle
        if obs % 2 == 0:
            # Hurdle: spans basically the entire width, with one random offset
            center_y = random.randint(m_to_idx(0.7), course_y_max - m_to_idx(0.7))
            y1 = max(0, center_y - hurdle_width // 2)
            y2 = min(course_y_max, center_y + hurdle_width // 2)
            x1 = min(x, course_x_max - hurdle_length)
            x2 = min(x + hurdle_length, course_x_max)

            height_field[x1:x2, y1:y2] = hurdle_height

            # Place goal some distance after obstacle (flat ground)
            goal_x = min(x2 + (gap_between // 2), course_x_max - 1)
            goals[obs + 1] = [goal_x, center_y]
            obstacle_indices.append(('hurdle', (x1, x2, y1, y2)))
            x = x2 + gap_between

            last_y = center_y

        else:
            # Balance beam: position the beam randomly left/right but within central corridor
            beam_center_y = random.randint(m_to_idx(1.1), course_y_max - m_to_idx(1.1))
            half_beam = beam_width // 2
            y1 = max(0, beam_center_y - half_beam)
            y2 = min(course_y_max, beam_center_y + half_beam)
            x1 = min(x, course_x_max - beam_length)
            x2 = min(x + beam_length, course_x_max)

            height_field[x1:x2, y1:y2] = beam_height

            # Place the goal at the far end of the beam (centered on beam)
            goal_x = min(x2 - 1, course_x_max - 1)
            goals[obs + 1] = [goal_x, beam_center_y]
            obstacle_indices.append(('beam', (x1, x2, y1, y2)))
            x = x2 + gap_between

            last_y = beam_center_y

    # Final goal at remaining flat area, at center y
    last_goal_x = min(course_x_max - m_to_idx(1.0), x)
    goals[-1] = [last_goal_x, last_y]
    height_field[x:, :] = 0  # Ensure final part is flat

    # (OPTIONAL: Pit for failed beam crossing? Skipped to emphasize balance, not jumps.)

    # Quick check: Make sure all goal indices are within field bounds
    goals = np.clip(goals, 0, np.array(height_field.shape) - 1)

    return height_field, goals