import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping-stone narrow beams for balance: a series of offset beams ('balance beams') across a pit, requiring the robot to balance, walk precise turns, and transition from one beam to the next."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # ---- Course Design ----
    #
    # The robot spawns on flat ground and must cross a "pit" (height -1.0) using a sequence of beams.
    # Each beam is 1-1.5m long, narrow (width is 0.3-0.5m), elevated at height 0.
    # The beams are offset/angled to force precise turning, with each goal at the end of a beam.
    # At higher difficulties, the beams become narrower and farther apart.
    #
    # Obstacle -- narrow beams, pit, turn positions.
    # Skill -- balance, precise foot placement, tight turns.
    #
    #          |=========|
    #          | pit     |
    #   spawn--====      |====      |====      |====    finish
    #          |  beam 1 |  beam 2 |  ...     |beam 4
    # 
    # The robot must not fall into the pit (height -1), and each beam sits at "ground level" (height 0).

    # ------- Parameters -------
    num_beams = 4
    spawn_length = 2.0  # flat ground at start
    finish_length = 1.0  # flat ground at end
    pit_height = -1.0

    total_length = m_to_idx(length)
    total_width = m_to_idx(width)

    beam_lengths = [
        1.1 - 0.2 * difficulty + random.uniform(-0.05, 0.05) for _ in range(num_beams)
    ]
    beam_widths = [
        max(0.3, 0.5 - 0.15 * difficulty + random.uniform(-0.02, 0.02)) for _ in range(num_beams)
    ]  # never less than 0.3m

    lateral_offsets = [
        random.uniform(-0.8, 0.8) * difficulty for _ in range(num_beams)
    ]  # beams shift left/right up to 0.8m at max difficulty

    spacing = [0.5 + 0.7 * difficulty + random.uniform(-0.07, 0.05) for _ in range(num_beams-1)]
    # distance (in m) "gap" from end of one beam to start of next

    # Center in y
    center_y = m_to_idx(width / 2)

    # Pre-populate height_field with pit from after spawn to before finish
    spawn_idx = m_to_idx(spawn_length)
    finish_idx = m_to_idx(length - finish_length)
    height_field[spawn_idx:finish_idx, :] = pit_height

    # Flat ground in spawn and finish
    height_field[:spawn_idx, :] = 0
    height_field[finish_idx:, :] = 0

    # -------------- Place Beams --------------
    x = spawn_idx
    y = center_y

    beam_centers = []

    for i in range(num_beams):
        l = beam_lengths[i]
        w = beam_widths[i]
        l_idx = m_to_idx(l)
        w_idx = max(1, m_to_idx(w))
        x_start = int(x)
        x_end = min(total_length, int(x + l_idx))
        # Offset in y for turning and balancing
        lateral = m_to_idx(lateral_offsets[i])
        y_center = int(np.clip(y + lateral, w_idx//2, total_width - 1 - w_idx//2))
        y_start = y_center - w_idx // 2
        y_end = y_center + (w_idx + 1)//2
        # Place beam (careful not to overflow boundaries)
        height_field[x_start:x_end, y_start:y_end] = 0.0  # ground level

        # Keep track of beam center for goal placement
        beam_centers.append(( (x_start + x_end) // 2, (y_start + y_end) // 2 ))

        # Next beam x position
        x = x_end + (m_to_idx(spacing[i]) if i < len(spacing) else 0)
        y = y_center

    # --------- Place Start/Finish Goals & Intermediates ----------
    # First goal at starting zone
    goals[0] = [m_to_idx(1), center_y]

    # Place one goal at the end of each beam (centered on the beam)
    for i, (gx, gy) in enumerate(beam_centers):
        goals[i+1] = [gx, gy]

    # In case num_beams<4, make sure to fill in all 5 goals
    for i in range(len(beam_centers)+1, 5):
        goals[i] = [finish_idx + m_to_idx(0.5), center_y]  # finish

    return height_field, goals