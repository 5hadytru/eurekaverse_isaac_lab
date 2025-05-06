import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of "see-saw" tilting balance beams the quadruped must cross, testing dynamic balancing."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Configuration
    random.seed(42)  # Ensures reproducibility
    np.random.seed(42)

    mid_y = m_to_idx(width // 2)
    spawn_length = m_to_idx(2)
    course_length = m_to_idx(length)
    course_width = m_to_idx(width)

    # See-saw dimensions
    num_seesaws = 6
    seesaw_length = 1.8 - 0.5 * difficulty   # meters; gets shorter at higher difficulty
    seesaw_length_idx = m_to_idx(seesaw_length)
    seesaw_width = 1.05 - 0.25 * difficulty  # meters; gets narrower at higher difficulty, but still >0.8 at max
    seesaw_width_idx = max(m_to_idx(seesaw_width), m_to_idx(0.8))
    seesaw_height = 0.12 + 0.12 * difficulty # meters; beam stands higher for more tip and challenge
    seesaw_angle_deg = 8 + 22 * difficulty   # Angle of tilt at rest, up to 30 degrees at max difficulty
    seesaw_angle_rad = np.deg2rad(seesaw_angle_deg)

    gap_length = 0.35 + 0.5 * difficulty     # meters between seesaws; forces robot to confidently step/stride
    gap_length_idx = m_to_idx(gap_length)

    # Width offset per seesaw
    y_offsets = np.linspace(-0.7, 0.7, num_seesaws)
    y_offsets = y_offsets * (1 - 0.5 * difficulty) # smaller offsets as difficulty increases

    # Keep spawn area flat
    height_field[0:spawn_length, :] = 0
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    # Make under-beam a pit
    height_field[spawn_length:, :] = -1.0  # Fill with pit

    # Place the seesaw beams
    cur_x = spawn_length
    for i in range(num_seesaws):
        x1 = int(cur_x)
        x2 = int(np.clip(x1 + seesaw_length_idx, x1+1, course_length))

        # Offset the seesaws a bit in y, but keep within bounds
        beam_center_y = mid_y + m_to_idx(y_offsets[i])
        y1 = int(np.clip(beam_center_y - seesaw_width_idx//2, 0, course_width-1))
        y2 = int(np.clip(beam_center_y + seesaw_width_idx//2, y1+1, course_width))

        # Simulate tilted seesaw: one end up, one down
        slope = np.tan(seesaw_angle_rad) * field_resolution  # meters rise per grid cell
        # left end lower, right end higher:
        for j, x in enumerate(range(x1, x2)):
            beam_height = -0.04 + (slope * j)
            beam_height = np.clip(beam_height, -0.04, seesaw_height)
            height_field[x, y1:y2] = beam_height

        # Put goal at seesaw midpoint
        goal_x = x1 + seesaw_length_idx // 2
        # Move the goal slightly forward at higher skill (makes robot slow down for beam end)
        if difficulty > 0.7:
            goal_x = x1 + int(0.7 * seesaw_length_idx)
        goals[i+1] = [goal_x, int((y1 + y2) // 2)]

        # Move to next seesaw: gap after the current one
        cur_x = x2 + gap_length_idx

    # Final solid ground area
    if cur_x < course_length:
        height_field[cur_x:, :] = 0
    # Final goal
    goals[7] = [min(course_length - m_to_idx(1), cur_x + m_to_idx(0.7)), mid_y]

    return height_field, goals