import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of narrow elevated balance beams running lengthwise, testing the robot's precise walking and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course plan:
    # - The robot starts on a flat platform.
    # - There are 5 long, narrow "balance beams" (ramps placed lengthwise).
    # - Each beam is separated by a short "drop zone" (pit), which requires careful stepping or minor jumps.
    # - The beams and pits shift left and right, so the robot must carefully steer to stay on.
    # - Final goal is on a wide safe platform.
    # Course forces the robot to keep precise balance and sometimes turn slightly, never descend lower than the spawn.
    # The "beams" are 0.32–0.45m wide. Their height above the ground increases with difficulty.
    # Drop/pit depth and gap width increase with difficulty.

    # --- Parameters (all relative to difficulty and robot size) ---
    spawn_pad_length = 1.2      # Robot starts on a 1.2m flat pad
    main_beam_length = 1.7      # Each main beam is ~1.7m long (so 5 fit before the 12m mark)
    final_pad_length = 1.2      # Exit area at end
    num_beams = 5
    # Beam width: narrow (0.32–0.45m)
    beam_width = 0.45 - 0.13 * difficulty
    beam_height = 0.08 + 0.22 * difficulty  # elevate beams
    pit_width = 0.20 + 0.35 * difficulty  # width increases with difficulty
    pit_depth = 0.09 + 0.35 * difficulty  # negative, deep enough to penalize falling

    # Some random left-right offset for each beam, as "turning" challenge:
    max_offset = (width/2 - beam_width/2 - 0.10)
    y_offsets = [np.random.uniform(-max_offset, max_offset) for _ in range(num_beams)]

    # Helper to put a rectangular region at a set height
    def set_region(x0, x1, y0, y1, h):
        height_field[x0:x1, y0:y1] = h

    # Convert measurements to indices
    spawn_pad_length_idx = m_to_idx(spawn_pad_length)
    main_beam_length_idx = m_to_idx(main_beam_length)
    pit_width_idx = m_to_idx(pit_width)
    final_pad_length_idx = m_to_idx(final_pad_length)
    beam_width_idx = m_to_idx(beam_width)
    pit_depth_val = -pit_depth  # negative height

    # Center along y-axis
    mid_y = m_to_idx(width / 2)
    half_beam = beam_width_idx // 2

    # 1. Set initial platform (flat)
    set_region(0, spawn_pad_length_idx, mid_y-half_beam-2, mid_y+half_beam+2, 0)  # slightly wider for launch
    goals[0] = [spawn_pad_length_idx//2, mid_y]

    x_pos = spawn_pad_length_idx

    # 2. Loop through beams and pits
    for beam_idx in range(num_beams):
        # Pick y-offset for this beam
        y_center = mid_y + m_to_idx(y_offsets[beam_idx])

        # a) Add beam
        set_region(
            x_pos, x_pos+main_beam_length_idx,
            y_center-half_beam, y_center+half_beam, beam_height
        )
        # Goal at center of this beam
        goals[beam_idx + 1] = [
            x_pos + main_beam_length_idx//2,
            y_center
        ]

        x_pos += main_beam_length_idx

        # b) Pit/gap after beam (except last one)
        if beam_idx < num_beams - 1:
            set_region(
                x_pos, x_pos+pit_width_idx,
                0, m_to_idx(width), pit_depth_val
            )
            x_pos += pit_width_idx

    # 3. Final platform ("safe zone")
    set_region(x_pos, x_pos+final_pad_length_idx, 0, m_to_idx(width), 0)
    goals[6] = [x_pos + final_pad_length_idx//2, mid_y]
    # Last goal: near exit
    goals[7] = [m_to_idx(length) - 2, mid_y]

    return height_field, goals