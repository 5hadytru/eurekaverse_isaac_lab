import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of balanced beams of varying width, alternating with open ground, to test the quadruped's precise foot placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course parameters
    beam_count = 5
    beam_length_m = 1.6 - 0.3 * difficulty
    beam_length = m_to_idx(beam_length_m)
    min_beam_width = 0.35
    max_beam_width = 0.6 - 0.25 * difficulty  # gets narrower at high difficulty, but always > 0.1
    space_between_beams = 0.5 + 0.4 * difficulty  # Make gaps a bit longer with difficulty
    space_between_beams = m_to_idx(space_between_beams)
    beam_height = 0.14 + 0.10 * difficulty      # simulate a raised beam (up to ~0.24m high)

    mid_y = m_to_idx(width / 2)

    # Leave spawn area untouched (flat ground at 0m height, width-full)
    spawn_length = m_to_idx(2)
    height_field[:spawn_length, :] = 0.0
    goals[0] = [m_to_idx(1.0), mid_y]          # First goal is shortly after spawn

    # For challenge, alternate beams slightly left and right
    lateral_offsets = np.linspace(-0.65, 0.65, beam_count)
    lateral_offsets = field_resolution * np.round(lateral_offsets / field_resolution)
    beam_locs = []
    cur_x = spawn_length

    for i in range(beam_count):
        # Beam width can reduce with difficulty, but always above minimum
        beam_width = np.clip(max_beam_width - 0.08*i*difficulty, min_beam_width, max_beam_width)
        half_width = m_to_idx(beam_width / 2)

        # Place beam centered, but alternate offset left/right from center
        offset = lateral_offsets[i % len(lateral_offsets)]
        y_center = mid_y + m_to_idx(offset)

        x_start = cur_x
        x_end = min(cur_x + beam_length, m_to_idx(length) - 1)
        y1 = max(y_center - half_width, 0)
        y2 = min(y_center + half_width, m_to_idx(width))

        # Make "open ground" (gap between beams) stay at 0m (walkable), so the robot must keep to the beam for efficient progress, but can step down with penalty if falls.
        # Optionally for high difficulty, make the ground in the gaps negative (e.g. -0.2), but always leave the last region flat so we don't trap the robot.

        # Raise the beam
        height_field[x_start:x_end, y1:y2] = beam_height

        # Add beam record and goal
        beam_locs.append((x_start, x_end, y_center))
        goal_x = (x_start + x_end) // 2
        goals[i+1] = [goal_x, y_center]
        
        # Add gap after beam (open ground)
        gap_start = x_end
        gap_end = min(gap_start + space_between_beams, m_to_idx(length) - 1)
        if i < beam_count - 1:
            if difficulty > 0.5:
                # At high difficulty, make the ground negative in the gaps for extra penalty (not a true pit, soft penalty).
                height_field[gap_start:gap_end, :] = -0.22 * difficulty
        cur_x = gap_end

    # Final "safe zone": flat ground till the end, for robot to stop
    height_field[cur_x:, :] = 0
    # Place the final goal just before course end, in the middle
    goals[6] = [min(cur_x + m_to_idx(0.4), m_to_idx(length) - 2), mid_y]
    goals[7] = [m_to_idx(length) - m_to_idx(0.5), mid_y]

    return height_field, goals