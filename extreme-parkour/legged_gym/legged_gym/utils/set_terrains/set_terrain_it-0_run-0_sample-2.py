import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of 'stair step' ledges: robot must climb up or down ledges, each stretching across the width."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters
    num_steps = 6  # six steps/ledges
    min_step_w = 1.0  # minimum ledge width, in meters
    max_step_w = 1.65  # maximum, keep a bit under 1/6th the course
    # Ledge widths and heights vary by difficulty
    base_step_w = np.linspace(min_step_w, max_step_w, num_steps) * (1 - 0.4 * difficulty)
    base_step_w = np.clip(base_step_w, min_step_w, max_step_w)
    step_gap = 0.5 + 0.7 * difficulty  # meters: distance between ledges
    step_gap_idx = m_to_idx(step_gap)
    ledge_height_min = 0.06 + 0.15 * difficulty  # meters, lowest ledge height
    ledge_height_max = 0.20 + 0.35 * difficulty  # meters, highest ledge height
    
    # Randomly decide if stairs go up, down, or alternate direction based on difficulty
    stair_type = random.choices(
        ['up', 'down', 'alternating', 'random'], 
        weights=[0.3, 0.3, 0.25, 0.15] if difficulty > 0.2 else [0.6, 0.3, 0.1, 0.0],
        k=1
    )[0]

    # Setup: spawn area
    spawn_x0 = 0
    spawn_x1 = m_to_idx(2.0)
    height_field[spawn_x0:spawn_x1, :] = 0
    mid_y = m_to_idx(width / 2)  # centerline

    # Place initial goal at spawn
    goals[0] = [spawn_x1 - m_to_idx(0.5), mid_y]

    # Begin ledge sequence after spawn area
    cur_x = spawn_x1

    # Track running elevation for each ledge
    elevation = 0.0
    elevation_dir = 1 if stair_type == 'up' else -1  # Up stairs or down stairs
    alternating = (stair_type == 'alternating')

    for step_id in range(num_steps):
        step_width = base_step_w[step_id]
        step_w_idx = m_to_idx(step_width)

        # Ledges always run fully across width (1m+ required)
        x0 = int(cur_x)
        x1 = int(cur_x + step_w_idx)
        y0 = 0
        y1 = m_to_idx(width)

        # Determine how high this ledge is compared to previous
        # Alternate/Random stairs can switch direction, otherwise monotonic
        if alternating and (step_id % 2 == 1):
            elevation_dir *= -1
        elif stair_type == 'random' and (random.random() < 0.3 + 0.55*difficulty):
            elevation_dir *= -1

        if elevation_dir > 0:
            ledge_height = random.uniform(ledge_height_min, ledge_height_max)
        else:
            ledge_height = -random.uniform(ledge_height_min, ledge_height_max)

        # For each ledge, add height to previous elevation, but clamp total elevation
        next_elevation = elevation + ledge_height
        max_elevation = 0.50 + 0.28 * difficulty
        min_elevation = -0.20 if difficulty > 0.15 else -0.07
        next_elevation = np.clip(next_elevation, min_elevation, max_elevation)
        ledge_height = next_elevation - elevation  # adjust actual step height

        elevation = next_elevation

        # Set ledge
        height_field[x0:x1, y0:y1] = elevation

        # Place gap after this ledge (unless final)
        gap_x0 = x1
        gap_x1 = int(x1 + step_gap_idx)
        if step_id < num_steps - 1:
            # Height in gap is either baseline or minimum previous height
            gap_height = min(0, elevation)  # allow step down, but no pit
            height_field[gap_x0:gap_x1, y0:y1] = gap_height
        cur_x = gap_x1

        # Place goal at center of each ledge
        if step_id < 7:  # up to 8 goals spaced out
            goal_x = (x0 + x1) // 2
            goals[step_id + 1] = [goal_x, mid_y]

    # Final approach to end
    x_end = m_to_idx(length)
    if cur_x < x_end:
        # Flat ground at last elevation for rest of course
        height_field[cur_x:x_end, :] = elevation
        # Final goal at end
        goals[-1] = [x_end - m_to_idx(0.5), mid_y]

    return height_field, goals