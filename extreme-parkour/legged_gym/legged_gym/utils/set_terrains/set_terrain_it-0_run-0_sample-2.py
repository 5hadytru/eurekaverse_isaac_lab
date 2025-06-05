import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Series of sloped ramps with narrow bridges in between to test balance, incline walking, and transitions."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((5, 2))

    # --- Course parameters ---
    # Central course axis, all obstacles centered in y
    mid_y = m_to_idx(width / 2)
    field_L = m_to_idx(length)
    field_W = m_to_idx(width)

    # Each segment: Ramp up (ascend), narrow bridge (flat), ramp down (descend), repeat

    # Ramps: longer/easier at low difficulty, steeper and shorter at high difficulty
    ramp_height = 0.18 + 0.26 * difficulty         # total up/down height (meters)
    ramp_length = 1.5 - 0.7 * difficulty           # meters
    ramp_len_idx = m_to_idx(ramp_length)
    ramp_height_idx = ramp_height                  # height in meters (no need to quantize)
    ramp_width = 1.3 - 0.5 * difficulty            # ramps get slightly narrower as difficulty increases
    ramp_W = m_to_idx(ramp_width)

    # Bridge: short, flat, and narrow, but always ≥0.45m wide
    bridge_length = 0.7 + 0.3 * difficulty         # meters
    bridge_len_idx = m_to_idx(bridge_length)
    bridge_width = max(0.45, 0.7 - 0.25 * difficulty)
    bridge_W = m_to_idx(bridge_width)

    # Safety margins
    safety_margin = m_to_idx(0.15)                 # always a small margin from field edge

    spawn_x = m_to_idx(1.0)
    # Make sure first ramp starts after safe area
    cur_x = max(m_to_idx(2.0), spawn_x + m_to_idx(0.2))

    # Start: Flat area for spawn, no obstacles
    height_field[:cur_x, :] = 0
    goals[0] = [spawn_x, mid_y]  # First goal is straight ahead from spawn

    # In total, fit 4 ramp-bridge segments in the 12 m course (5th goal is end of last ramp)
    ramp_bridge_segs = 4
    segs = []
    for i in range(ramp_bridge_segs):
        # Compute y-center for this segment; allow a mild zig-zag with up to ±0.5m offset
        seg_y_offset = int(round((random.uniform(-0.5, 0.5) * (1-difficulty)) * (field_W / width)))

        # --- Ascending ramp ---
        ramp_start_x = cur_x
        ramp_end_x = ramp_start_x + ramp_len_idx
        y1 = max(safety_margin, mid_y + seg_y_offset - ramp_W // 2)
        y2 = min(field_W - safety_margin, mid_y + seg_y_offset + ramp_W // 2)

        # Linear ramp up in x-axis
        for xi in range(ramp_start_x, ramp_end_x):
            rel = (xi - ramp_start_x) / max(1, (ramp_end_x - ramp_start_x - 1))
            height_field[xi, y1:y2] = rel * ramp_height_idx

        # --- Flat, narrow bridge at top ---
        bridge_start_x = ramp_end_x
        bridge_end_x = bridge_start_x + bridge_len_idx
        by1 = max(safety_margin, mid_y + seg_y_offset - bridge_W // 2)
        by2 = min(field_W - safety_margin, mid_y + seg_y_offset + bridge_W // 2)
        height_field[bridge_start_x:bridge_end_x, by1:by2] = ramp_height_idx
        # Set pit under and around bridge (using negative heights), except bridge zone
        pit_depth = -0.25 - 0.25 * difficulty
        # To ensure robot stays on bridge, create pit wider than bridge:
        pit_W = int(round(1.0 + 1.0 * difficulty) / field_resolution)
        pit_y1 = max(0, int((by1 + by2) / 2) - pit_W // 2)
        pit_y2 = min(field_W, int((by1 + by2) / 2) + pit_W // 2)
        height_field[bridge_start_x:bridge_end_x, :pit_y1] = pit_depth
        height_field[bridge_start_x:bridge_end_x, pit_y2:] = pit_depth

        # --- Descending ramp ---
        rampd_start_x = bridge_end_x
        rampd_end_x = rampd_start_x + ramp_len_idx
        for xi in range(rampd_start_x, rampd_end_x):
            rel = 1 - (xi - rampd_start_x) / max(1, (rampd_end_x - rampd_start_x - 1))
            height_field[xi, y1:y2] = rel * ramp_height_idx

        # Set segment for goal, place in center of bridge (goal 1-4)
        if i < 3:
            goal_x = bridge_start_x + bridge_len_idx // 2
            goal_y = (by1 + by2) // 2
            goals[i+1] = [goal_x, goal_y]

        segs.append((ramp_start_x, ramp_end_x, bridge_start_x, bridge_end_x, rampd_start_x, rampd_end_x, y1, y2, by1, by2))
        # Next segment starts after downward ramp, with a small buffer
        buffer = m_to_idx(0.18 + 0.2 * difficulty)
        cur_x = rampd_end_x + buffer

    # Set final (5th) goal at end of last down-ramp, centered
    last_seg = segs[-1]
    final_goal_x = min(field_L-1, (last_seg[5] + m_to_idx(0.5)))
    final_goal_y = (last_seg[6] + last_seg[7]) // 2
    goals[4] = [final_goal_x, final_goal_y]

    # Set any space after last ramp to level ground at zero
    height_field[cur_x:, :] = 0

    return height_field, goals