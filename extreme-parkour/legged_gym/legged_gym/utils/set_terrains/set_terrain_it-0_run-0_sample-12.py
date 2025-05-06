import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of balance beams of variable width, length, and side shift, requiring crossing precision and lateral stepping."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for beams
    min_beam_width = 0.40  # meter (narrow but passable)
    max_beam_width = 1.0   # meter
    min_beam_length = 1.2
    max_beam_length = 2.0

    min_gap = 0.20     # in meters; robot will need to step across, not a jump.
    max_gap = 0.55 + 0.45 * difficulty  # at hardest, close to max step reach

    beam_height = 0.12 + 0.08 * difficulty  # raise beam with difficulty

    n_beams = 6
    mid_y = m_to_idx(width / 2)
    cur_x = m_to_idx(2) # start after spawning zone

    # Start with a checkerboard pit for penalizing falls
    height_field[mid_x_start := cur_x:, :] = -1.2  # pit everywhere except on beams

    def add_beam(start_x, end_x, center_y, beam_width, h):
        """Places a beam of given width and height onto the terrain."""
        half_w = m_to_idx(beam_width / 2)
        x1, x2 = int(start_x), int(end_x)
        y1 = int(max(0, center_y - half_w))
        y2 = int(min(m_to_idx(width), center_y + half_w))
        height_field[x1:x2, y1:y2] = h

    # Spawn platform (flat region before first beam)
    spawn_length = m_to_idx(2)
    height_field[0:spawn_length, :] = 0  # safe landing area for spawn

    # Place first goal at spawn
    goals[0] = [spawn_length-m_to_idx(0.5), mid_y]

    # Arrange beams with varying width, length, and y-position
    y_center = mid_y
    for i in range(n_beams):
        beam_length = np.random.uniform(min_beam_length, max_beam_length)
        beam_width = np.random.uniform(
            min_beam_width, 
            min_beam_width + (max_beam_width - min_beam_width) * (1 - difficulty)
        )  # narrower at higher difficulty

        beam_length_idx = m_to_idx(beam_length)
        beam_width_idx = beam_width  # still meters, for helper
        beam_gap = np.random.uniform(min_gap, max_gap)
        beam_gap_idx = m_to_idx(beam_gap)

        # Lateral offset for each beam, so the robot must steer left and right
        max_shift = 1.3 * (difficulty)
        y_shift = np.random.uniform(-max_shift, max_shift)
        y_center = int(np.clip(y_center + m_to_idx(y_shift), m_to_idx(beam_width/2), m_to_idx(width - beam_width/2) - 1))

        add_beam(cur_x, cur_x + beam_length_idx, y_center, beam_width, beam_height)

        # Place a goal in the middle of the current beam
        goals[i+1] = [cur_x + beam_length_idx // 2, y_center]
        # Advance x by beam_length + gap
        cur_x += beam_length_idx + beam_gap_idx

    # Final landing platform (flat ground)
    final_platform_len = m_to_idx(1.2)
    height_field[cur_x:cur_x+final_platform_len, :] = 0

    # Last goal near the end of the course
    if len(goals) < 8 or goals.shape[0] < 8:
        raise RuntimeError("goals must be of length 8")
    goals[7] = [cur_x + final_platform_len//2, mid_y]

    return height_field, goals