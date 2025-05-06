import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A zigzag balance beam course testing the quadruped's balance and agility."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Configuration: Zigzag balance beams over a pit (all negative below beams)
    # The beam difficulty is scaled by width and zigzag angle
    beam_width = 0.35 + (1.0 - difficulty) * 0.35  # from 0.7m (easiest) to 0.35m (hardest)
    beam_width_idx = m_to_idx(beam_width)
    beam_height = 0.25 + difficulty * 0.1           # between 0.25-0.35m over pit
    pit_depth =  -0.7 - 0.3 * difficulty            # deeper pit with difficulty (-0.7m to -1m)
    beam_length = 1.4 + 1.0 * difficulty            # beams become longer, 1.4m->2.4m
    beam_length_idx = m_to_idx(beam_length)
    beam_turn_angle = np.deg2rad(18 + 50 * difficulty) # angle of each zigzag (18-68°)
    num_beams = 5

    total_course_length_idx = m_to_idx(length)
    course_width_idx = m_to_idx(width)

    # Set pit over (x > 2m), keep 2m of spawn area at 0 height
    spawn_x_idx = m_to_idx(2)
    height_field[spawn_x_idx:,:] = pit_depth

    # Central placement, so beams zigzag left & right about course center
    mid_y_idx = course_width_idx // 2

    # Helper for placing rectangular beam
    def place_beam(center_x_idx, center_y_idx, angle_rad, beam_length_idx, beam_width_idx, height):
        """Places a rectangular beam patch into the height_field."""
        l_half = beam_length_idx // 2
        w_half = beam_width_idx // 2
        # For each grid point within an expanded bounding box, check if in rotated beam rectangle
        for dx in range(-l_half-2, l_half+3):
            for dy in range(-w_half-2, w_half+3):
                # Rotate point (dx, dy) by -angle
                x_off = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
                y_off = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
                # Check if inside beam rectangle
                if abs(x_off) <= l_half and abs(y_off) <= w_half:
                    xi = center_x_idx + int(np.round(dx))
                    yi = center_y_idx + int(np.round(dy))
                    if 0 <= xi < total_course_length_idx and 0 <= yi < course_width_idx:
                        height_field[xi,yi] = height

    # Build zigzag beam course and assign goals at each beam center and turn
    # Initial beam is straight along course; each beam alternates zig direction
    beam_centers = []
    x = spawn_x_idx + beam_length_idx//2 + 2
    y = mid_y_idx
    angle = 0  # radians, initial: straight ahead
    sign = 1
    for i in range(num_beams):
        # Place beam
        place_beam(x, y, angle, beam_length_idx, beam_width_idx, beam_height)
        beam_centers.append((x, y))
        # Next goal: at center of beam
        if i < 7:  # Store up to 8 goals
            goals[i] = [x, y]
        # For next beam: compute new x,y
        dx = int(np.round(np.cos(angle) * (beam_length_idx + m_to_idx(0.2))))
        dy = int(np.round(np.sin(angle) * (beam_length_idx + m_to_idx(0.2))))
        # For zig, alternate sign each time and calculate new angle
        angle += sign * beam_turn_angle
        sign *= -1  # flip direction
        x = x + dx
        y = y + dy
        # Clamp position to field boundaries (avoid going out at extreme angle/difficulty)
        x = np.clip(x, 0, total_course_length_idx-1)
        y = np.clip(y, m_to_idx(1.0), course_width_idx-m_to_idx(1.0))  # beams always at least 1m from edge

    # Place final goal at flat ground off-pit (rides up a ramp)
    ramp_length = m_to_idx(1.0)
    ramp_x0 = min(total_course_length_idx - ramp_length, x)
    ramp_x1 = min(total_course_length_idx, x + ramp_length)
    height_field[ramp_x0:ramp_x1, :] = np.linspace(beam_height, 0, ramp_x1-ramp_x0)[:, None]
    goals[7] = [ramp_x1-1, mid_y_idx]

    # Place first goal at start (spawn)
    start_goal_x = m_to_idx(1.0)
    goals[0] = [start_goal_x, mid_y_idx]

    return height_field, goals