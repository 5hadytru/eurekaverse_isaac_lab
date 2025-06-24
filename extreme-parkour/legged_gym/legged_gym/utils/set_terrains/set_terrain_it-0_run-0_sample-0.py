import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping-stone logs: Robot must cross a series of narrow rolling log beams spaced over a shallow pit."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for the logs (cylindrical "beams" to step across)
    log_count = 6
    # Difficulty increases gap length and log narrowness
    min_log_width = 0.35   # must be >=0.28 (robot width)
    log_width = np.clip(0.7 - 0.3*difficulty, min_log_width, 1.4)
    log_length = width - 0.2   # logs span most of width
    log_height = 0.12 + 0.13 * difficulty  # higher logs for higher difficulty
    min_gap = 0.10
    max_gap = 0.52 + 0.45*difficulty
    gap = np.linspace(min_gap, max_gap, log_count+1)  # gaps get wider

    pit_depth = -0.5 - 0.7*difficulty   # deeper pit on higher difficulty

    spawn_length = m_to_idx(2)
    mid_y = m_to_idx(width/2)
    field_L = m_to_idx(length)
    field_W = m_to_idx(width)

    # Set starting area as flat ground
    height_field[0:spawn_length, :] = 0

    # Start first goal in spawn area
    goals[0] = [m_to_idx(1), mid_y]

    # Set the pit after the spawn (except where logs will be placed)
    height_field[spawn_length:, :] = pit_depth

    # Calculate log positions
    log_positions = []
    x = spawn_length + m_to_idx(gap[0])  # first log after spawn
    log_x_centers = []

    for i in range(log_count):
        log_c = int(x)
        log_x_centers.append(log_c)
        log_L = m_to_idx(log_length)/2
        log_W = m_to_idx(log_width)/2

        # For realism, leave 10 cm from edges
        y_start = m_to_idx(0.10)
        y_end = field_W - m_to_idx(0.10)

        # Place the log as a rectangular bump
        height_field[log_c-m_to_idx(0.25):log_c+m_to_idx(0.25), y_start:y_end] = log_height

        # Center each goal on the log
        goals[i+1] = [log_c, mid_y]

        # Step to next position
        x += m_to_idx(0.45) + m_to_idx(gap[i+1]) + random.randint(-2, 2)  # some noise

    # The final section is a flat "exit" area
    exit_start = min(m_to_idx(length)-m_to_idx(1), log_x_centers[-1]+m_to_idx(0.70))
    height_field[exit_start:, :] = 0
    goals[-1] = [exit_start + m_to_idx(0.5), mid_y]

    return height_field, goals