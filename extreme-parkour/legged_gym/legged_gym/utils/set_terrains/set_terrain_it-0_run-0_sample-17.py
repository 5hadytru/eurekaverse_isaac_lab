import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A slalom of elevated balance beams testing precise, narrow-footing walking."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for beams
    # Beams become longer, narrower, and higher with difficulty
    base_beam_length = 2.0 + difficulty * 2.0        # meters
    min_beam_width = 0.4
    max_beam_width = 0.7 - 0.2 * difficulty          # narrower at high difficulty
    beam_widths = np.linspace(max_beam_width, min_beam_width, 4)
    beam_height = 0.1 + 0.25 * difficulty            # meters above ground
    beam_spacing = 1.0 + 0.6 * difficulty            # meters between beams
    approach_clear = 2.0                             # meters flat ground at start
    exit_clear = 1.0                                 # meters flat after last beam

    mid_y = m_to_idx(width // 2)
    field_len = m_to_idx(length)
    field_wid = m_to_idx(width)

    # Set spawn/approach/exit zone as flat ground
    height_field[:m_to_idx(approach_clear), :] = 0
    height_field[-m_to_idx(exit_clear):, :] = 0

    # Mark spawn goal
    goals[0] = [m_to_idx(approach_clear*0.75), mid_y]

    cur_x = m_to_idx(approach_clear)

    # Number of beams (4) and 2 beams per leg (one left, one right alternately)
    for i in range(4):
        # Alternate beam position left/right of centerline for slalom
        side_sign = -1 if i % 2 == 0 else 1
        beam_y_center = mid_y + side_sign * m_to_idx(0.67)  # shift left/right by 2x robot width
        beam_width = m_to_idx(beam_widths[i])

        # Beam start/end index
        beam_len_idx = m_to_idx(base_beam_length + 0.4 * (random.random()-0.5)) # ±0.2m random
        beam_x1 = cur_x
        beam_x2 = beam_x1 + beam_len_idx

        # Ensure beam fits in field
        if beam_x2 + m_to_idx(beam_spacing) > field_len - m_to_idx(exit_clear):
            beam_x2 = field_len - m_to_idx(exit_clear)
            beam_len_idx = beam_x2 - beam_x1

        # Position beam across width (centered on beam_y_center)
        beam_y1 = max(0, beam_y_center - beam_width//2)
        beam_y2 = min(field_wid, beam_y_center + beam_width//2)

        # Raise beam to desired height
        height_field[beam_x1:beam_x2, beam_y1:beam_y2] = beam_height

        # Set goal at the end of each beam, middle of the beam
        # Move y slightly ahead/off to force the robot to re-center for next beam
        goal_x = beam_x2 - m_to_idx(0.3)  # a bit back from the very end
        goal_y = int((beam_y1 + beam_y2) // 2)
        goals[i+1] = [goal_x, goal_y]

        # Add a pit (lower field) between beams, except after last
        pit_x1 = beam_x2
        pit_x2 = min(field_len, beam_x2 + m_to_idx(beam_spacing))
        if i < 3:
            height_field[pit_x1:pit_x2, :] = -0.5 - 0.5 * difficulty  # deepens with hardness

        cur_x = pit_x2

    # Place intermediate goals on flat ground at the end
    for j in range(5, 8):
        # Spread them out in the approach/exit clear area
        x = field_len - m_to_idx(exit_clear) + (j-4)*(m_to_idx(exit_clear)//4)
        x = min(x, field_len-1)
        goals[j] = [x, mid_y]

    return height_field, goals