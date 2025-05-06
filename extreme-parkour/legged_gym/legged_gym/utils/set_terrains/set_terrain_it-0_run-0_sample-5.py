import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Alternating narrow balance beams and wide stepping stones over a deep trench to test balance and precise foot placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course settings
    mid_y = m_to_idx(width // 2)
    spawn_length = m_to_idx(2)  # robot spawns with center at x=1, so 2m of safe ground
    field_x, field_y = m_to_idx(length), m_to_idx(width)
    
    # --- Parameters based on difficulty ---
    beam_w = 0.5 - 0.2 * difficulty   # Narrower beam at higher difficulty [0.5m to 0.3m]
    beam_w = max(beam_w, 0.3)
    beam_l = 1.6 + 0.2 * difficulty   # Slightly longer beam at higher difficulty [1.6m to 1.8m]
    beam_h = 0.08 + 0.22 * difficulty # Higher beam at higher difficulty [0.08 to 0.3m]
    stone_r = 0.6 - 0.18 * difficulty # Stepping stones smaller at high difficulty [0.6 to 0.42m]
    stone_h = 0.06 + 0.14 * difficulty # Stones a bit taller at higher difficulty
    trench_depth = -1.0
    gap_l = 0.55 + 0.3 * difficulty   # Gap between obstacles; longer at higher difficulty [0.55 to 0.85m]

    # Quantize
    beam_w_idx = m_to_idx(beam_w)
    beam_l_idx = m_to_idx(beam_l)
    beam_h_f = beam_h # Will be used as height, not index
    stone_r_idx = m_to_idx(stone_r)
    stone_h_f = stone_h
    gap_l_idx = m_to_idx(gap_l)

    # -- Flat ground in spawn area --
    height_field[0:spawn_length, :] = 0
    # Put first goal at spawn center
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    # -- Make everything after spawn a trench --
    height_field[spawn_length:, :] = trench_depth

    # -- Layout: Alternate beams and stepping stones; 3 beams, 3 stones, then goal --
    cur_x = spawn_length
    margin_y = m_to_idx(0.4) # margin from edges so obstacles are always fully within course
    num_obstacles = 6 # 3 beams + 3 stones
    for i in range(num_obstacles):
        if i % 2 == 0:
            # Balance beam
            # Beams alternate offset to right/left to force turning slightly
            direction = -1 if (i // 2) % 2 == 0 else 1 
            beam_mid_y = mid_y + direction * m_to_idx(0.7 * difficulty) # At higher diff, shift more
            
            x1 = cur_x
            x2 = min(x1 + beam_l_idx, field_x-1)
            y1 = max(beam_mid_y - beam_w_idx//2, margin_y)
            y2 = min(beam_mid_y + beam_w_idx//2, field_y-margin_y)
            height_field[x1:x2, y1:y2] = beam_h_f

            # Put goal at center of beam
            gx = (x1 + x2) // 2
            gy = (y1 + y2) // 2
            goals[i+1] = [gx, gy]
            obs_len = beam_l_idx

        else:
            # Stepping stone (circular platform)
            stone_x = min(cur_x + stone_r_idx + 2, field_x-1)
            # Stagger stone slightly in y
            stone_y = mid_y + m_to_idx(random.uniform(-0.3, 0.3) * (1 + difficulty)) 
            stone_x1 = max(stone_x - stone_r_idx, 0)
            stone_x2 = min(stone_x + stone_r_idx, field_x-1)
            stone_y1 = max(stone_y - stone_r_idx, margin_y)
            stone_y2 = min(stone_y + stone_r_idx, field_y-margin_y)

            # Draw round stone in x1:x2, y1:y2
            for xx in range(stone_x1, stone_x2):
                for yy in range(stone_y1, stone_y2):
                    # Circle equation
                    if ((xx-stone_x)**2 + (yy-stone_y)**2) <= stone_r_idx**2:
                        height_field[xx, yy] = stone_h_f
            
            # Set goal on center of stone
            goals[i+1] = [stone_x, stone_y]
            obs_len = 2*stone_r_idx

        # Move to next obstacle, include a gap
        cur_x = int(cur_x + obs_len + gap_l_idx)

    # -- Final stretch: flat, with goal at end --
    end_x = min(field_x-1, cur_x + m_to_idx(1.0))
    height_field[cur_x:end_x, :] = 0
    goals[7] = [int((cur_x + end_x) // 2), mid_y]

    return height_field, goals