import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of raised, narrow, turning beams over a deep pit to test precise foot placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Robot parameters
    robot_length, robot_width = 0.645, 0.28

    # Beam parameters scaling with difficulty
    min_beam_width = max(0.45, robot_width + 0.15 - 0.1 * difficulty)
    max_beam_width = max(0.7, robot_width + 0.6 - 0.3 * difficulty)
    beam_length = 1.7 - 0.7 * difficulty  # beams shorter as difficulty increases
    min_beam_height = 0.18 + 0.21 * difficulty
    max_beam_height = 0.4 + 0.22 * difficulty

    beam_overlap = 0.16 - 0.09 * difficulty  # beams overlap slightly for easier transitions
    beam_pitch_angle = np.deg2rad(25 + 50 * difficulty)  # max angular offset between beams

    n_beams = 7  # 7 beams, 8 goals including spawn and end
    mid_y = m_to_idx(width) // 2

    spawn_x = m_to_idx(1)
    spawn_y = mid_y
    spawn_length = m_to_idx(2)

    # Set flat ground at spawn
    height_field[:spawn_length, :] = 0

    # Set the pit in the rest of terrain (deepest at higher difficulty)
    pit_depth = -1.0 - 1.5 * difficulty
    height_field[spawn_length:, :] = pit_depth

    # First goal: spawn
    goals[0] = [spawn_x, spawn_y]

    # Beam placement
    x, y = m_to_idx(2.4), mid_y
    current_angle = 0
    np.random.seed(42)  # deterministic for debugging

    for i in range(n_beams):
        # Beam geometry
        this_beam_width = m_to_idx(np.random.uniform(min_beam_width, max_beam_width))
        this_beam_length = m_to_idx(beam_length + np.random.uniform(-0.13, 0.13))
        this_beam_height = np.random.uniform(min_beam_height, max_beam_height)
        # Angle change, to make a beam "turn"
        if i > 0:
            turn = np.random.choice([-1, 1]) * np.random.uniform(0.2, beam_pitch_angle if i < n_beams-1 else 0.14)
        else:
            turn = 0
        current_angle += turn

        dx = int(np.cos(current_angle) * this_beam_length)
        dy = int(np.sin(current_angle) * this_beam_length)
        cx, cy = x, y

        # Draw the beam by filling rectangles along orientation (simple approach)
        for t in range(this_beam_length):
            bx = int(cx + np.cos(current_angle) * t)
            by = int(cy + np.sin(current_angle) * t)
            wx1, wx2 = bx - this_beam_width//2, bx + this_beam_width//2
            wy1, wy2 = by - this_beam_width//2, by + this_beam_width//2
            # Bounds-check
            if 0 <= wx1 < height_field.shape[0] and 0 <= wx2 < height_field.shape[0] and \
               0 <= wy1 < height_field.shape[1] and 0 <= wy2 < height_field.shape[1]:
                height_field[wx1:wx2, wy1:wy2] = this_beam_height

        # Set goal in the middle of this beam
        beam_goal_x = int(cx + np.cos(current_angle) * (this_beam_length // 2))
        beam_goal_y = int(cy + np.sin(current_angle) * (this_beam_length // 2))
        if i < goals.shape[0]-1:
            goals[i+1] = [beam_goal_x, beam_goal_y]

        # Advance tip for next beam (add some overlap so the robot can step across)
        x = int(cx + np.cos(current_angle) * (this_beam_length - m_to_idx(beam_overlap)))
        y = int(cy + np.sin(current_angle) * (this_beam_length - m_to_idx(beam_overlap)))

    # Last goal at flat area at far end
    final_flat_x = m_to_idx(length) - m_to_idx(1)
    final_flat_y = mid_y
    height_field[final_flat_x:, :] = 0.0
    goals[-1] = [final_flat_x, final_flat_y]

    # (Optional) Clamp height_field so no values exceed spawn or sink further
    height_field = np.clip(height_field, pit_depth, np.max(height_field))

    return height_field, goals