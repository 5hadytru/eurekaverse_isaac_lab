import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A series of sloped ramps and steps in alternating directions to test the quadruped's uphill and downhill traversal ability."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((4, 2))

    # Parameters based on quadruped size
    ramp_length = max(1.4, 2.5 - 1.0 * difficulty)           # Each ramp covers > 2 body lengths
    ramp_height = 0.12 + 0.22 * difficulty                   # Max ramp height rises with difficulty
    ramp_width = max(1.0, 1.7 - 0.3 * difficulty)            # Make slightly narrower at harder levels
    step_height = 0.04 + 0.11 * difficulty                   # Steps at ramp tops, low at easy difficulty
    step_length = 0.4 + 0.25 * difficulty                    # Steps are just longer than quadruped

    ramp_length_idx = m_to_idx(ramp_length)
    ramp_width_idx = m_to_idx(ramp_width)
    ramp_height = float(ramp_height)
    step_length_idx = m_to_idx(step_length)
    step_height = float(step_height)

    course_length_idx = m_to_idx(length)
    course_width_idx = m_to_idx(width)
    mid_y = course_width_idx // 2

    safe_zone_idx = m_to_idx(2)
    # Keep the spawn area flat
    height_field[:safe_zone_idx, :] = 0
    goals[0, :] = [m_to_idx(1), mid_y]  # Start goal

    # Each obstacle direction (ramp up, step, ramp down, etc.) - four obstacles/zones
    x_ptr = safe_zone_idx
    for i in range(3):
        # Alternate ramp orientation: up, down, up
        up = (i % 2 == 0)

        # Keep ramps at least 1m wide, randomize slight lateral offset within bounds
        y_offset = random.randint(-m_to_idx(0.5), m_to_idx(0.5))
        y1 = max(0, mid_y - ramp_width_idx//2 + y_offset)
        y2 = min(course_width_idx, y1 + ramp_width_idx)
        # Ensure y bounds are valid
        if y2 > course_width_idx:
            y1 -= (y2 - course_width_idx)
            y2 = course_width_idx

        # RAMP: linear slope up or down
        x_end_ramp = min(x_ptr + ramp_length_idx, course_length_idx)
        ramp_y_region = slice(y1, y2)
        
        if up:
            height_start = height_field[x_ptr-1, mid_y] if x_ptr > 0 else 0
            for j, x in enumerate(range(x_ptr, x_end_ramp)):
                val = height_start + ramp_height * (j / max(1, ramp_length_idx-1))
                height_field[x, ramp_y_region] = val
            ramp_top = height_start + ramp_height
        else:
            height_start = height_field[x_ptr-1, mid_y] if x_ptr > 0 else ramp_height
            for j, x in enumerate(range(x_ptr, x_end_ramp)):
                val = height_start - ramp_height * (j / max(1, ramp_length_idx-1))
                height_field[x, ramp_y_region] = val
            ramp_top = height_start - ramp_height

        # STEP: short block after ramp to break gait and require stepping up/down (like a curb)
        x1_step = x_end_ramp
        x2_step = min(x1_step + step_length_idx, course_length_idx)
        # Slightly randomize step width but keep safe min
        w_step = max(m_to_idx(1.0), ramp_width_idx - m_to_idx(0.15))
        y1_step = max(0, mid_y - w_step//2 + y_offset)
        y2_step = min(course_width_idx, y1_step + w_step)
        if y2_step > course_width_idx:
            y1_step -= (y2_step - course_width_idx)
            y2_step = course_width_idx

        if up:
            height_field[x1_step:x2_step, y1_step:y2_step] = ramp_top + step_height
            step_top = ramp_top + step_height
        else:
            height_field[x1_step:x2_step, y1_step:y2_step] = ramp_top - step_height
            step_top = ramp_top - step_height

        # Set goals on top of steps, center in y direction
        goals[i+1] = [x1_step + (x2_step - x1_step) // 2, (y1_step + y2_step)//2]

        # Advance pointer past step, add short flat landing area before next ramp
        x_ptr = x2_step + m_to_idx(0.2)
        # Add a small flat area between step and next ramp for transition
        if x_ptr < course_length_idx:
            y_flat1 = max(0, mid_y - m_to_idx(0.7) + y_offset)
            y_flat2 = min(course_width_idx, y_flat1 + m_to_idx(1.4))
            height_field[x2_step:x_ptr, y_flat1:y_flat2] = step_top

    # Flat finish region (at goal)
    flat_end = min(x_ptr + m_to_idx(1.2), course_length_idx)
    height_field[x_ptr:flat_end, :] = height_field[x_ptr-1, mid_y] if x_ptr > 0 else 0
    # Final goal is in the center near far end
    goals[3, :] = [flat_end - m_to_idx(0.8), mid_y]

    return height_field, goals