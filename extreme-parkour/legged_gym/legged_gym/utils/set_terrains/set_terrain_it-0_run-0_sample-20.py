import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A zigzag elevated balance beam ('rail walk'), with gaps and turns above a pit, testing dynamic balance and precise turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Initialize terrain to ground level
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    ###### BEAM AND PIT PARAMETERS ######
    beam_height = 0.15 + 0.15 * difficulty  # meters above pit bottom
    pit_depth = 0.3 + 0.5 * difficulty  # how deep the pit is (NEGATIVE height, everywhere except beam)
    beam_width = 0.28 + 0.12 * (1-difficulty)    # from 40 cm (easy) to 28 cm (hard) at robot's own width
    beam_width = max(beam_width, 0.23)           # safety: never < robot's width
    beam_length = 1.7 - 0.5 * difficulty # Each beam section
    gap_len = 0.2 + 0.3 * difficulty
    turn_angle_deg = 15 + 40 * difficulty    # How sharp can a segment turn? 15deg to 55deg
    mid_y = m_to_idx(width/2)
    spawn_length = m_to_idx(2)

    # Set spawn area to regular ground
    height_field[0:spawn_length, :] = 0
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    # Terrain after spawn is ALL pit at -depth, except where filled by a beam
    height_field[spawn_length:, :] = -pit_depth

    ###### BEAM ZIGZAG #######
    x = float(spawn_length)
    y = mid_y
    angle = 0.0 # Horizontal

    for i in range(1, 8):  # 7 beam segments, 8 goals
        # 1. Calculate this segment's direction (a left or right turn at random, not always straight)
        if i == 1:
            dtheta = 0 # first always straight
        else:
            dtheta = np.deg2rad(random.choice([-1, 1]) * (turn_angle_deg * random.uniform(0.7, 1.0)))
            angle += dtheta

        # 2. Compute this beam's start/end in (x, y) field coords (as floats)
        seg_len = beam_length * random.uniform(0.88, 1.12)
        x1_float, y1_float = x, y
        x2_float = x1_float + seg_len * np.cos(angle)
        y2_float = y1_float + seg_len * np.sin(angle)

        # Ensure this beam stays within bounds
        x2_float = np.clip(x2_float, 0, m_to_idx(length)-1)
        y2_float = np.clip(y2_float, m_to_idx(beam_width/2), m_to_idx(width) - m_to_idx(beam_width/2) - 1)

        # 3. Draw rectangle of width beam_width along the line
        Np = int(np.linalg.norm([x2_float-x1_float, y2_float-y1_float])) + 1
        for p in range(Np):
            t = p / max(Np-1, 1)
            xi = int(np.round(x1_float * (1-t) + x2_float * t))
            yi = int(np.round(y1_float * (1-t) + y2_float * t))
            y_lo = int(yi - m_to_idx(beam_width/2))
            y_hi = int(yi + m_to_idx(beam_width/2)) + 1
            if xi >= 0 and xi < height_field.shape[0]:
                height_field[xi, y_lo:y_hi] = beam_height

        # 4. Place a gap after the beam (pit continues through gap)
        x_gap = x2_float + m_to_idx(gap_len)
        # Set goal at end of the current beam (centered)
        goals[i] = [x2_float, y2_float]
        x, y = x_gap, y2_float # start next beam after gap

        # Stop if we reached the end
        if x >= m_to_idx(length)-3:
            x = m_to_idx(length)-2  # place last goal within bounds
            break

    # Make sure last goal is inside terrain
    goals[-1,0] = min(goals[-1,0], m_to_idx(length)-2)
    goals[-1,1] = np.clip(goals[-1,1], m_to_idx(beam_width/2), m_to_idx(width)-m_to_idx(beam_width/2)-1)

    # Ensure goals are valid indices (clip for safety)
    goals = np.clip(goals, 0, np.array([height_field.shape[0]-1, height_field.shape[1]-1]))

    return height_field, goals