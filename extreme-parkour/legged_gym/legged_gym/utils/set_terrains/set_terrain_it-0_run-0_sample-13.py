import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Slalom course with alternating wide step-over barriers that force zig-zag lateral turns."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters
    spawn_x = m_to_idx(1.0)
    mid_y = m_to_idx(width / 2)
    n_barriers = 6  # 6 alternating barriers to generate 7 weaving segments
    barrier_width = m_to_idx(1.00)  # the barriers run the full width, but the path is cut out
    barrier_thickness = m_to_idx(0.20)  # thickness along the x-axis
    min_channel_w = 1.1 - 0.6*difficulty  # meters, 1.1m (easy) to 0.5m (hard)
    min_channel_w = max(0.5, min_channel_w)
    channel_width = m_to_idx(min_channel_w)
    barrier_height = 0.14 + 0.21*difficulty  # from 0.14m (easy) to 0.35m (hard)
    
    # Compute barrier positions
    margin_x = m_to_idx(1.0)  # Give at least 1 m before first barrier after spawn
    start_x = spawn_x + margin_x
    dx = (m_to_idx(length) - start_x - m_to_idx(1.0)) // (n_barriers+1)
    x_positions = [int(start_x + (i+1)*dx) for i in range(n_barriers)]

    # Y-positions for open channels (alternate sides)
    # On each barrier, leave a channel at either hard left or hard right.
    edge_padding = m_to_idx(0.15)
    left_channel_center  = edge_padding + channel_width//2
    right_channel_center = m_to_idx(width) - edge_padding - channel_width//2

    # Sides alternate for each barrier: left, right, left, right...
    channel_centers = [left_channel_center if i%2==0 else right_channel_center for i in range(n_barriers)]

    # Spawn/initial goal
    goals[0] = [spawn_x//2, mid_y]

    prev_goal_x = spawn_x
    prev_goal_y = mid_y

    for i, (x_b, y_c) in enumerate(zip(x_positions, channel_centers)):
        # Place barrier
        y_start = 0
        y_end = m_to_idx(width)
        x1 = x_b
        x2 = min(x1 + barrier_thickness, m_to_idx(length))

        # Clear channel in barrier at correct side
        c_center = int(y_c)
        c_half = channel_width // 2

        # Set the whole barrier
        height_field[x1:x2, y_start:y_end] = barrier_height

        # Cut out a channel
        y1 = max(int(c_center - c_half), 0)
        y2 = min(int(c_center + c_half), m_to_idx(width))
        height_field[x1:x2, y1:y2] = 0.0  # clear out the path

        # Set goal just past the channel to force zig-zag
        # Place it halfway beyond the barrier and centered in the open channel
        next_goal_x = min(x2 + m_to_idx(0.30), m_to_idx(length)-1)  # move 0.3m forward from barrier
        next_goal_y = c_center
        goals[i+1] = [next_goal_x, next_goal_y]

        prev_goal_x = next_goal_x
        prev_goal_y = next_goal_y

    # Final straight goal at the end:
    final_goal_x = m_to_idx(length) - m_to_idx(0.50)
    # Final goal alternates again
    final_goal_y = left_channel_center if n_barriers%2==0 else right_channel_center
    goals[-1] = [final_goal_x, final_goal_y]

    # Keep the final bit flat to let the robot finish
    height_field[final_goal_x:, :] = 0

    return height_field, goals