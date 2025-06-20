import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Logs and ramps: The robot traverses a series of log-like cylindrical obstacles and angled ramps that require climbing, balancing, and stepping down."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # Basic parameters
    course_length = m_to_idx(length)
    course_width = m_to_idx(width)
    mid_y = course_width // 2

    # 1. Start area flat and safe
    spawn_length = m_to_idx(2.0)
    height_field[:spawn_length, :] = 0

    # 2. First section: "Log" balance (force straight walking and coordination)
    # The log is a narrow but traversable round obstacle crossing the path in x direction for at least 1.0 m wide
    log_start = m_to_idx(2.2)
    log_length = m_to_idx(1.5 + 1.0 * difficulty)  # longer logs as difficulty increases
    log_width = m_to_idx(1.0)
    log_radius = 0.07 + 0.12 * difficulty  # higher means harder
    log_center_y = mid_y + m_to_idx(random.uniform(-0.5, 0.5))  # add slight offset

    # Add "log" using a rounded top
    for x in range(log_start, log_start + log_length):
        for y in range(log_center_y - log_width // 2, log_center_y + log_width // 2):
            # Calculate distance from center for the rounded effect
            rel = (y - log_center_y) * field_resolution
            if abs(rel) < log_radius:
                height_field[x, y] = np.sqrt(log_radius ** 2 - rel ** 2)
            else:
                height_field[x, y] = -0.1   # small depression/flanking sides so it can't bypass easily

    # First goal after the log
    goals[0] = [spawn_length, mid_y]  # spawn line
    goals[1] = [log_start + log_length // 2, log_center_y]

    # 3. Second section: Steep up ramp (bilateral leg strength & climbing)
    ramp1_start = log_start + log_length + m_to_idx(0.2)
    ramp1_length = m_to_idx(1.0 + 0.5 * difficulty)
    ramp1_width = m_to_idx(1.4)
    ramp1_height = 0.1 + 0.18 * difficulty  # makes ramp steeper/higher on harder settings

    ramp1_center_y = mid_y + m_to_idx(random.uniform(-0.2, 0.2))
    ramp1_y1 = ramp1_center_y - ramp1_width // 2
    ramp1_y2 = ramp1_center_y + ramp1_width // 2

    for i, x in enumerate(range(ramp1_start, ramp1_start + ramp1_length)):
        slope = (i / ramp1_length) * ramp1_height
        height_field[x, ramp1_y1:ramp1_y2] = slope

    # 2nd goal: Halfway up the ramp
    halfway_ramp_x = ramp1_start + ramp1_length // 2
    goals[2] = [halfway_ramp_x, ramp1_center_y]

    # 4. Third section: Ramp down (descending control and safe stepping)
    ramp2_start = ramp1_start + ramp1_length
    ramp2_length = m_to_idx(1.0 + 0.4 * difficulty)
    ramp2_width = m_to_idx(1.4)
    ramp2_height = ramp1_height  # Same height as up ramp
    ramp2_center_y = mid_y + m_to_idx(random.uniform(-0.2, 0.2))
    ramp2_y1 = ramp2_center_y - ramp2_width // 2
    ramp2_y2 = ramp2_center_y + ramp2_width // 2

    for i, x in enumerate(range(ramp2_start, ramp2_start + ramp2_length)):
        slope = ramp2_height * (1 - i / ramp2_length)
        height_field[x, ramp2_y1:ramp2_y2] = slope

    # 3rd goal: End of the down ramp
    ramp2_exit_x = ramp2_start + ramp2_length - 1
    goals[3] = [ramp2_exit_x, ramp2_center_y]

    # 5. Fourth section: Alternating short "logs" as stepping stones (precise stepping and planning)
    stones_start = ramp2_start + ramp2_length + m_to_idx(0.2)
    num_stones = 3 + int(2 * difficulty)
    stone_spacing = m_to_idx(0.5 + 0.4 * difficulty)  # spread out more at harder settings
    stone_width = m_to_idx(0.45)
    stone_length = m_to_idx(0.50)
    stone_height = 0.10 + 0.13 * difficulty
    stone_y_offsets = [m_to_idx(d) for d in np.linspace(-0.4, 0.4, num_stones) if abs(d) < (course_width // 2 - stone_width)]

    for i in range(num_stones):
        cx = stones_start + i * stone_spacing
        cy = mid_y + random.choice([-1, 1]) * (random.randint(0, m_to_idx(0.8)))
        cy = np.clip(cy, stone_width, course_width - stone_width)
        xf, xt = int(cx - stone_length // 2), int(cx + stone_length // 2)
        yf, yt = int(cy - stone_width // 2), int(cy + stone_width // 2)
        height_field[xf:xt, yf:yt] = stone_height

    # Penultimate goal: On last stone
    last_stone_x = stones_start + (num_stones - 1) * stone_spacing
    last_stone_y = mid_y
    goals[4] = [last_stone_x, last_stone_y]

    # After last goal: last several meters are flat for smooth finish
    end_flat_start = last_stone_x + m_to_idx(0.7)
    height_field[end_flat_start:, :] = 0

    # Clamp everything to fit inside the field bounds and not overflow
    height_field = height_field[:course_length, :course_width]
    goals = np.clip(goals, [[0, 0]], [[course_length-1, course_width-1]])

    return height_field, goals