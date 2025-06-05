import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of alternating ramps and sidewise balance beams to challenge robot stability and precision."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Core parameters
    mid_y = m_to_idx(width/2)
    field_len = m_to_idx(length)
    field_wid = m_to_idx(width)

    # --- Obstacle dimensions ---
    # Ramp variables
    min_ramp_height = 0.10 + 0.10 * difficulty    # meters
    max_ramp_height = 0.30 + 0.25 * difficulty    # meters
    ramp_length = 1.25 - 0.50 * difficulty        # slightly longer, shorter at high diff.
    ramp_width = 1.2                              # meters, always >= 1 meter
    
    # Beam variables
    beam_length = 1.1 + 0.5 * difficulty          # meters; longer beam at higher difficulty
    beam_width = 0.40 + 0.15 * (1-difficulty)     # meters; wider beam at lower diff, narrower at high
    beam_height = 0.18 + 0.18 * difficulty        # meters (robot can't just step off easily)
    
    # Offsets and counts
    spawn_x = m_to_idx(2)                         # Leave spawn area clear!
    n_obstacles = 3                               # 3 ramps, 3 beams = 6 obstacles + start & end
    safe_margin = m_to_idx(0.3)
    obstacle_gap = m_to_idx(0.30 + 0.40 * difficulty)  # Between obstacles

    # Helper for inserting ramp
    def add_ramp(x0, y0, ramp_len, ramp_wid, h0, h1):
        x0, y0 = int(x0), int(y0)
        x1 = x0 + m_to_idx(ramp_len)
        y1 = max(0, y0 - m_to_idx(ramp_wid/2))
        y2 = min(field_wid, y0 + m_to_idx(ramp_wid/2))
        for xi in range(x0, min(field_len, x1)):
            advance = (xi - x0) / max(1, x1-x0-1)
            h = (1-advance)*h0 + advance*h1
            height_field[xi, y1:y2] = h
    
    # Helper for inserting beam (straight, sideways for fun)
    def add_beam(x0, y0, beam_len, beam_wid, beam_h, angle_deg=0):
        # Beam extends along angle from (x0, y0)
        angle = np.deg2rad(angle_deg)
        dx = np.cos(angle)
        dy = np.sin(angle)
        nsteps = m_to_idx(beam_len)
        bwid = m_to_idx(beam_wid)
        for step in range(nsteps):
            xi = int(round(x0 + dx*step))
            yi_c = int(round(y0 + dy*step))
            y1 = max(0, yi_c - bwid//2)
            y2 = min(field_wid, yi_c + bwid//2)
            if 0 <= xi < field_len:
                height_field[xi, y1:y2] = beam_h

    # Initial flat spawn
    height_field[:spawn_x, :] = 0.0
    goals[0] = [spawn_x-m_to_idx(0.5), mid_y]

    cur_x = spawn_x
    for i in range(n_obstacles):
        # RAMP
        rh0 = 0.0
        rh1 = np.random.uniform(min_ramp_height, max_ramp_height)
        # Place ramp straight, left-to-right
        add_ramp(cur_x, mid_y, ramp_length, ramp_width, rh0, rh1)
        # Goal top of ramp
        ramp_goal_x = cur_x + m_to_idx(ramp_length * 0.8)
        goals[1+i*2] = [ramp_goal_x, mid_y]
        cur_x += m_to_idx(ramp_length) + safe_margin

        # BEAM
        # Beams alternate offset left/right of center
        beam_angle = 8 * (1 if i%2==0 else -1)       # Slight beam angle, increases challenge
        # Beam lateral offset at higher diff
        beam_center_y = mid_y + int((i%2)*2-1) * int(beam_width*1.2 * (0.25 + 0.75*difficulty))
        add_beam(cur_x, beam_center_y, beam_length, beam_width, beam_height, angle_deg=beam_angle)
        # Next goal: center of beam
        beam_goal_x = cur_x + m_to_idx(beam_length*0.5)
        goals[2+i*2] = [beam_goal_x, beam_center_y]
        cur_x += m_to_idx(beam_length) + obstacle_gap

    # Final goal, flat ground after last obstacle
    final_x = min(cur_x + m_to_idx(0.8), field_len - 1)
    height_field[cur_x:, :] = 0.0
    goals[7] = [final_x, mid_y]

    # Clamp all goals to inside height_field
    for i in range(8):
        goals[i,0] = min(max(0, goals[i,0]), height_field.shape[0]-1)
        goals[i,1] = min(max(0, goals[i,1]), height_field.shape[1]-1)

    return height_field, goals