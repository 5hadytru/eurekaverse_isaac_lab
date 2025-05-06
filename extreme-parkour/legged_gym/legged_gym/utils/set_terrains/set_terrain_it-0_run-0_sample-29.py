import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of balance beams and narrow walkways for testing precise foot-placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters
    mid_y = m_to_idx(width) // 2
    spawn_length_idx = m_to_idx(2)
    length_idx = m_to_idx(length)
    width_idx = m_to_idx(width)

    # 1. Flat spawn area
    height_field[:spawn_length_idx, :] = 0
    goals[0] = [m_to_idx(1.0), mid_y]  # First goal in spawn flat ground

    # 2. Balance beam section (robot must traverse a narrow raised walkway)
    beam_length = 1.7 + difficulty * 2.1      # (1.7m to 3.8m)
    beam_width = 0.25 + 0.15 * (1-difficulty) # (0.4m at easy, 0.25m at hard)
    beam_height = 0.12 + 0.12 * difficulty    # (0.12m to 0.24m)
    beam_gap = 0.22 + 0.45 * difficulty       # (0.22m to 0.67m)
    # Stagger/yaw beam slightly for higher difficulty
    beam_angle = (0.0 if difficulty < 0.3 
                  else np.random.uniform(-0.12, 0.12) * difficulty)

    cur_x = spawn_length_idx
    for sec in range(3):  # 3 balance beams, separated by gaps
        bl = m_to_idx(beam_length)
        bw = m_to_idx(beam_width)
        bh = beam_height
        gap = m_to_idx(beam_gap)

        # Beam center (add a small random offset to test recovery)
        center_y = mid_y + int(np.random.uniform(-2, 2) * difficulty)
        # Angle/yaw: walk diagonally for skill
        for b in range(bl):
            # Calculate offset if there is a yaw angle
            y_offset = int(b * np.tan(beam_angle))
            y1 = center_y - bw // 2 + y_offset
            y2 = center_y + bw // 2 + y_offset
            # Safety: stay within bounds (no overlap with wall)
            y1 = max(0, y1)
            y2 = min(width_idx, y2)
            height_field[cur_x+b, y1:y2] = bh

        # Goal at center of beam
        goals[sec+1] = [cur_x + bl // 2, center_y]
        cur_x += bl

        # Add a pit/gap after beam
        if cur_x + gap < length_idx:
            height_field[cur_x:cur_x+gap, :] = -0.8   # pit is down 0.8m

        cur_x += gap
        # Slightly change beam parameters for next one
        beam_length += 0.2 * (-1)**(sec) * difficulty
        beam_width = max(0.22, beam_width - 0.04 * difficulty)  # Keep safe range

    # 3. Wide stepping stone - to test transition from narrow to wide surface
    stone_length = m_to_idx(0.95 + 1.1 * difficulty)   # 0.95-2.05m
    stone_width = m_to_idx(1.0 + 0.5 * difficulty)     # 1-1.5m
    stone_height = 0.12 + 0.13 * difficulty
    stone_y = mid_y - stone_width // 2
    height_field[cur_x:cur_x+stone_length, stone_y:stone_y+stone_width] = stone_height
    goals[4] = [cur_x + stone_length//2, mid_y]
    cur_x += stone_length

    # 4. Slalom narrow walkway (zigzag beams)
    slalom_count = 2 + int(difficulty * 2)
    walkway_length = m_to_idx(0.63 + 0.37 * difficulty) # 0.63 to 1.0 meters per segment
    walkway_width = m_to_idx(0.28 + 0.06 * (1-difficulty)) # 0.28-0.34 meters
    walkway_height = 0.14 + 0.06 * difficulty
    slalom_offset_max = m_to_idx(1.10 * difficulty)     # how much left/right

    # Place alternating slalom beams
    for s in range(slalom_count):
        delta_y = ((-1)**s) * int(slalom_offset_max * np.random.uniform(0.8, 1.0))
        center_y = mid_y + delta_y
        y1 = max(0, center_y - walkway_width//2)
        y2 = min(width_idx, center_y + walkway_width//2)
        lx = min(walkway_length, length_idx - cur_x)
        height_field[cur_x:cur_x+lx, y1:y2] = walkway_height
        goals[5+s] = [cur_x + lx//2, center_y]
        cur_x += lx

        # Intersperse small gaps for difficulty
        if cur_x + m_to_idx(0.2) < length_idx:
            height_field[cur_x:cur_x+m_to_idx(0.2), :] = -0.6
            cur_x += m_to_idx(0.2)

    # 5. Final platform to land (for finish)
    platform_length = m_to_idx(1.0)
    height_field[cur_x:cur_x+platform_length, :] = 0.0
    # Last goal
    goals[7] = [min(cur_x + platform_length//2, length_idx-1), mid_y]

    # Clip all goals to be within field bounds
    goals = np.clip(goals, [0,0], [length_idx-1, width_idx-1])

    return height_field, goals