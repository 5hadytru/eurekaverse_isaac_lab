
import numpy as np
import random

def set_terrain(terrain, variation, difficulty):
    terrain_fns = [
        set_terrain_0,
        set_terrain_1,
        set_terrain_2,
        set_terrain_3,
        set_terrain_4,
        set_terrain_5,
        set_terrain_6,
        set_terrain_7,
        set_terrain_8,
        set_terrain_9,
        # INSERT TERRAIN FUNCTIONS HERE
    ]
    idx = int(variation * len(terrain_fns))
    height_field, goals = terrain_fns[idx](terrain.width * terrain.horizontal_scale, terrain.length * terrain.horizontal_scale, terrain.horizontal_scale, difficulty)
    terrain.height_field_raw = (height_field / terrain.vertical_scale).astype(np.int16)
    terrain.goals = goals
    return idx

def set_terrain_0(length, width, field_resolution, difficulty):
    """Stepping-stone 'urban blocks' course: jumping/walking atop sequential narrow rectangular blocks, testing balance and narrow foothold navigation."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- Parameters ---
    # Block and gap sizes
    block_width = 0.5 + 0.2 * (1 - difficulty)    # 0.7m at easy, 0.5m at hard
    block_length = 1.2 - 0.3 * difficulty         # 1.2m at easy, 0.9m at hard (always > robot length)
    gap_min = 0.15 + 0.20 * difficulty            # minimum 0.15m gap at easy, up to 0.35m+ at hard
    gap_max = gap_min + 0.2 * difficulty          # more variability with difficulty

    block_height = 0.15 + 0.23 * difficulty       # Easy: 0.15m, Hard: 0.38m (max ~robot's knee)
    pit_depth = 0.60 + 0.4 * difficulty           # Deep pits

    # Convert to indices
    block_width_i = m_to_idx(block_width)
    block_length_i = m_to_idx(block_length)
    gap_min_i, gap_max_i = m_to_idx(gap_min), m_to_idx(gap_max)
    block_height = float(block_height)
    pit_height = -float(pit_depth)

    margin_y = m_to_idx(0.5)   # keep all blocks away from edge of field

    # --- Place initial flat ground for spawn area ---
    spawn_length = m_to_idx(2)
    height_field[:spawn_length, :] = 0
    mid_y = m_to_idx(width / 2)

    # --- Construction loop ---
    cur_x = spawn_length
    # Stride pattern: most blocks straight, sometimes slight left/right offset
    y_positions = [mid_y]
    n_blocks = 7

    # First goal is in spawn region
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    for i in range(n_blocks):
        # Slight y offset: occasionally make the robot shift left/right, but not edge to edge
        if i > 0:
            if random.random() < 0.45:
                dy = m_to_idx(random.choice([-0.5, 0.5]))
                new_y = min(max(y_positions[-1] + dy,
                                margin_y + block_width_i//2),
                            m_to_idx(width) - margin_y - block_width_i//2)
                y_positions.append(int(new_y))
            else:
                y_positions.append(y_positions[-1])

        y_c = y_positions[i]
        x1 = cur_x
        x2 = min(x1 + block_length_i, m_to_idx(length) - 1)

        # Carve a pit around the block first
        pit_margin = m_to_idx(0.05)
        pit_x1 = max(x1 - pit_margin, spawn_length)
        pit_x2 = min(x2 + pit_margin, m_to_idx(length) - 1)
        pit_y1 = max(y_c - block_width_i//2 - pit_margin, 0)
        pit_y2 = min(y_c + block_width_i//2 + pit_margin, m_to_idx(width) - 1)
        height_field[pit_x1:pit_x2, pit_y1:pit_y2] = pit_height

        # Add the block
        b_y1 = int(y_c - block_width_i // 2)
        b_y2 = int(y_c + (block_width_i + 1) // 2)
        height_field[x1:x2, b_y1:b_y2] = block_height

        # Place a goal approximately at the center of this block
        goals[i+1] = [int((x1 + x2) // 2), int((b_y1 + b_y2) // 2)]

        # Next block: advance x by block + random gap
        gap = random.randint(gap_min_i, gap_max_i)
        cur_x += block_length_i + gap

        # Avoid placing blocks beyond the field
        if cur_x + block_length_i >= m_to_idx(length) - 1:
            break

    # Final goal: flat finishing area at exit
    fin_margin = m_to_idx(1)
    finish_x1 = min(cur_x, m_to_idx(length) - fin_margin)
    height_field[finish_x1:, :] = 0
    goals[7] = [min(finish_x1 + m_to_idx(0.5), m_to_idx(length) - 1), mid_y]

    return height_field, goals

def set_terrain_1(length, width, field_resolution, difficulty):
    """Series of balance beams and zig-zag turns testing lateral stability and precise turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for the balance beam course
    # Beams are narrow, so width 0.4-0.5m, length 2m. At higher difficulty, narrower and higher off ground.
    min_beam_width = 0.5 - 0.25 * difficulty     # 0.5m (easy) -> 0.25m (hard)
    beam_width = max(min_beam_width, 0.24)       # ensure no less than 0.24m (minimum allowed, a bit wider than robot body)
    beam_length = 2.0 - 0.25 * difficulty        # 2.0m -> 1.75m, shorter at high difficulty for more direction changes
    beam_height = 0.10 + 0.20 * difficulty       # 0.10m (easy) -> 0.30m (hard, can't just walk off/on)
    gap_between_beams = 0.10 + 0.25 * difficulty  # 0.1m (easy) -> 0.35m (hard)
    spawn_length = m_to_idx(2.0)
    mid_y = m_to_idx(width/2)

    n_beams = 4  # Zig-zag: 4 straight beams, robot zig-zags left/right
    beam_dirs = [0, 1, 0, -1]  # alternate straight, right, straight, left
 
    # Place spawn area
    height_field[:spawn_length, :] = 0
    goals[0] = [spawn_length//2, mid_y]  # first goal is at the spawn

    cur_x = spawn_length
    cur_y = mid_y
    beam_w = m_to_idx(beam_width)
    beam_l = m_to_idx(beam_length)
    beam_h = beam_height
    gap = m_to_idx(gap_between_beams)
    zig_offset = m_to_idx( (1.1 - 0.4*difficulty) )  # how far sideways to offset at the zig/zag (easy: 1.1m, hard: 0.7m)

    for i in range(n_beams):
        # Zig or zag
        if beam_dirs[i] == 0:
            # Straight
            next_y = cur_y
        else:
            next_y = cur_y + beam_dirs[i]*zig_offset
            # Clamp to within field
            next_y = np.clip(next_y, m_to_idx(beam_width)//2, m_to_idx(width) - m_to_idx(beam_width)//2 - 1)

        # The beam runs from cur_x to cur_x+beam_l, at current y, with width beam_w
        y1 = int(np.clip(cur_y - beam_w//2, 0, m_to_idx(width)-1))
        y2 = int(np.clip(cur_y + beam_w//2, 0, m_to_idx(width)-1))
        x1 = int(cur_x)
        x2 = int(np.clip(cur_x + beam_l, 0, m_to_idx(length)-1))
        height_field[x1:x2, y1:y2] = beam_h

        # Put goal in center of this beam
        goals[i+1] = [ (x1 + x2)//2, (y1 + y2)//2 ]

        # Update position: "hop" gap ahead; set next beam's center to new y
        cur_x = x2 + gap
        cur_y = next_y

        # Add a "pit" between beams so robot cannot drop down and re-climb
        pit_depth = -0.8 - 0.5*difficulty  # deep pit to force balancing
        pit_x1 = x2
        pit_x2 = int(np.clip(cur_x, 0, m_to_idx(length)-1))
        height_field[pit_x1:pit_x2, :] = pit_depth

    # Add a final wide (1.5m) landing platform at the end
    platform_length = m_to_idx(1.2)
    platform_width = m_to_idx(1.5)
    x1 = int(cur_x)
    x2 = int(np.clip(cur_x + platform_length, 0, m_to_idx(length)))
    y1 = int(np.clip(cur_y - platform_width//2, 0, m_to_idx(width)-1))
    y2 = int(np.clip(cur_y + platform_width//2, 0, m_to_idx(width)-1))
    height_field[x1:x2, y1:y2] = 0
    goals[5] = [ (x1 + x2)//2, (y1 + y2)//2 ]

    # The rest of the goals orient the finish--finish is simply at end platform far edge
    for g in range(6, 8):
        # Spread last two goals out along the wide end platform for straight finish
        gx = int( x2 - (g-5)*m_to_idx(0.2) )
        gy = (y1+y2)//2
        goals[g] = [gx, gy]

    # If needed, pad in the final goal in case not filled
    if (n_beams+1) < 8:
        for g in range(n_beams+1, 8):
            goals[g] = goals[n_beams]

    return height_field, goals

def set_terrain_2(length, width, field_resolution, difficulty):
    """Zig-zag ramp gauntlet: Multiple wide ramps at sharp angles requiring climbing, descending, and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))
    # Ramp properties scale with difficulty
    total_length_idx = m_to_idx(length)
    total_width_idx = m_to_idx(width)
    spawn_x_idx = m_to_idx(1)
    mid_y_idx = total_width_idx//2
    ramp_length_m = 2.2 - 0.5 * difficulty   # ramp too steep if too short!
    ramp_width_m  = 1.2 + 0.8 * difficulty   # wider at high diff: more options, less side-stepping at low diff
    ramp_height   = 0.12 + 0.23 * difficulty # higher at higher difficulty: 12cm to 35cm

    ramp_length = m_to_idx(ramp_length_m)
    ramp_width  = m_to_idx(ramp_width_m)

    turn_offset = m_to_idx(1.0) # how far 'vertically' the ramp shifts per segment

    # set spawn area clear
    height_field[:spawn_x_idx+1,:] = 0

    # Place first goal: start, unshifted
    cur_x = spawn_x_idx
    cur_y = mid_y_idx
    goals[0] = [cur_x, cur_y]

    directions = [+1, -1] * 4  # left, right, left, right, ... (up to 8 segments)

    # Main zig-zag ramp loop
    for i in range(7):
        # Compute bounding box for the ramp
        x0 = cur_x
        y0 = cur_y
        x1 = min(cur_x + ramp_length, total_length_idx - 1)
        # Side offset (zig/zag)
        dir = directions[i]
        y1 = np.clip(cur_y + int(dir * turn_offset), m_to_idx(0.5), total_width_idx-m_to_idx(0.5))

        # The ramp is a rectangle between y0 and y1, interpolating in y as it advances in x
        ramp_min_y = int(min(y0, y1) - ramp_width//2)
        ramp_max_y = int(max(y0, y1) + ramp_width//2)
        ramp_min_y = max(ramp_min_y, 0)
        ramp_max_y = min(ramp_max_y, total_width_idx-1)
        
        # Draw the ramp as a sloped plane (linear in x) from (cur_x, y0) to (x1, y1)
        # Each position (x, y) on the ramp gets an interpolated y center and a corresponding height
        for xi in range(x0, x1):
            frac = (xi - x0) / max(x1 - x0, 1)
            y_center = int(round((1-frac)*y0 + frac*y1))
            ramp_y_start = max(y_center - ramp_width//2, 0)
            ramp_y_end   = min(y_center + ramp_width//2, total_width_idx-1)
            ramp_h = ramp_height * frac # start of ramp is 0m, end is full height

            # Make the ramp rise then flatten then descend in next section
            if i % 2 == 0:
                # Ascend
                height_field[xi, ramp_y_start:ramp_y_end] = ramp_h
            else:
                # Descend
                height_field[xi, ramp_y_start:ramp_y_end] = ramp_height - ramp_h

        # Update next goal: at end of ramp in center
        next_x = x1
        next_y = int(round(y1))
        goals[i+1] = [next_x, next_y]
        cur_x, cur_y = next_x, next_y

    # Final flat goal at the end, set as last goal
    end_x = min(cur_x+m_to_idx(1), total_length_idx-1)
    goals[-1] = [end_x, cur_y]
    # Flat area at finish
    height_field[end_x:, :] = 0

    return height_field, goals

def set_terrain_3(length, width, field_resolution, difficulty):
    """Alternating balance beams with narrow planks over shallow pits: tests accurate foot placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Key parameters based on quadruped size and difficulty
    # Plank: balance beam height increases and gets more narrow with difficulty
    # Pit: gets deeper and wider as difficulty increases
    plank_length = 1.5  # meters
    plank_width = max(0.35, 0.8 - 0.4 * difficulty)  # meters, never smaller than 0.35m
    plank_height = 0.12 + 0.25 * difficulty        # meters
    pit_depth = -0.10 - 0.25 * difficulty          # meters, negative for pits
    pit_length = 0.6 + 0.8 * difficulty            # meters
    # Always leave spawn area flat
    spawn_length = 2.0  # meters

    # Place features in the X direction, alternating: [flat] → plank → [pit] → plank → [pit]...
    cur_x = m_to_idx(spawn_length)
    mid_y = m_to_idx(width / 2)
    beams = 6  # number of balance beams

    # Helper to add a plank at position
    def add_plank(center_x, mid_y):
        half_len = m_to_idx(plank_length / 2)
        half_width = m_to_idx(plank_width / 2)
        x1 = max(0, center_x - half_len)
        x2 = min(m_to_idx(length), center_x + half_len)
        y1 = max(0, mid_y - half_width)
        y2 = min(m_to_idx(width), mid_y + half_width)
        height_field[x1:x2, y1:y2] = plank_height

    # Helper to add a pit under and to the sides of the plank
    def add_pit(center_x, mid_y):
        half_len = m_to_idx(pit_length / 2)
        # The pit is wider than the plank, so the flanks are pits, but the plank remains above, flush with surface
        pit_margin = m_to_idx(0.05)  # leave small overlap for realism at plank edge
        y1 = 0
        y2 = m_to_idx(width)
        pit_x1 = max(0, center_x - half_len)
        pit_x2 = min(m_to_idx(length), center_x + half_len)
        # Under the plank, only set pit outside a little margin below plank
        plank_half_width = m_to_idx(plank_width / 2)
        # Left of plank
        height_field[pit_x1:pit_x2, y1:mid_y - plank_half_width - pit_margin] = pit_depth
        # Right of plank
        height_field[pit_x1:pit_x2, mid_y + plank_half_width + pit_margin:y2] = pit_depth

    # Set spawn area flat
    height_field[:cur_x, :] = 0.0
    # Place the first goal just ahead of spawn
    goals[0] = [cur_x - m_to_idx(0.7), mid_y]

    # Parameter for placing beams and pits
    step_dist = m_to_idx(1.4)  # nominal distance from beam center to center; tune as needed
    zigzag_offset = m_to_idx(0.55 + 0.3 * difficulty)  # how much to zigzag laterally (difficulties 0-1: 0.55--0.85m)

    # Alternate left/right zig-zag for each beam
    for i in range(beams):
        # Compute lateral offset
        if i % 2 == 0:
            beam_y = mid_y - zigzag_offset
        else:
            beam_y = mid_y + zigzag_offset
        # Add pit beneath and flanking where the plank will be
        add_pit(cur_x, beam_y)
        # Add the plank over the pit
        add_plank(cur_x, beam_y)
        # Place a goal just after each plank's end (so robot must travel straight over beam)
        if i < beams - 1:
            # Project a bit past the end of this beam, at the same y
            goals[i+1] = [cur_x + m_to_idx(plank_length / 2) - m_to_idx(0.1), beam_y]
        else:
            # For last beam, project closer to final flat area at course end
            goals[i+1] = [cur_x + m_to_idx(plank_length / 2), beam_y]
        # Advance cur_x for the next beam/pit
        cur_x += step_dist

    # Fill the region after the last beam to finish flat
    height_field[cur_x:, :] = 0.0
    # Make final goal at far end center
    goals[7] = [m_to_idx(length) - m_to_idx(0.8), mid_y]

    return height_field, goals

def set_terrain_4(length, width, field_resolution, difficulty):
    """Stepping stone balance beams: a zig-zagging series of narrow elevated beams testing lateral precision and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))  # 8 sequential waypoints

    # Parameters for stepping beams
    beam_length = 1.8 - 0.6 * difficulty  # beams get shorter as difficulty increases
    beam_length = max(beam_length, 0.8)
    beam_length_idx = m_to_idx(beam_length)
    beam_width = 0.45 + 0.2 * (1-difficulty)  # narrower beams as difficulty increases, never below 0.45m
    beam_width = max(beam_width, 0.4)
    beam_width_idx = m_to_idx(beam_width)
    beam_height = 0.08 + 0.18 * difficulty   # up to 26cm elevation

    gap_length = 0.27 + 0.4 * difficulty     # gaps between beams
    gap_length_idx = m_to_idx(gap_length)

    # Zig-zag offset amplitude (how far beams deviate left/right)
    zigzag_amp = 0.5 + 0.9 * difficulty     # up to ~1.4m for hard tasks (still must stay within bounds)
    zigzag_amp_idx = m_to_idx(zigzag_amp)
    zigzag_sign = 1

    # Centerline index
    mid_y = m_to_idx(width/2)

    # Reserve spawn area as flat ground (from 0 to x=2m)
    spawn_length = m_to_idx(2)
    height_field[0:spawn_length, :] = 0

    # Place first goal at the start of the first beam, just ahead of spawn area
    cur_x = spawn_length
    cur_y = mid_y
    goals[0] = [spawn_length - m_to_idx(0.5), cur_y]

    for i in range(7):  # 7 beams, last goal off the beams
        # Zig-zag in the y direction
        zigzag = m_to_idx(zigzag_sign * random.uniform(0.3, zigzag_amp))
        # Ensure beam always stays within bounds
        beam_y_center = max(min(cur_y + zigzag, m_to_idx(width) - beam_width_idx//2 - 1), beam_width_idx//2)
        zigzag_sign *= -1  # Alternate sides

        # Define beam rectangle
        x1 = cur_x
        x2 = min(cur_x + beam_length_idx, m_to_idx(length)-1)
        y1 = int(beam_y_center - beam_width_idx//2)
        y2 = int(beam_y_center + beam_width_idx//2)
        # Place beam: raised above the ground
        height_field[x1:x2, y1:y2] = beam_height

        # Place goal at beam center (slightly forward to keep moving)
        goals[i+1] = [x1 + beam_length_idx//2, beam_y_center]

        # Next beam: add gap
        cur_x = x2 + gap_length_idx
        # Also move y-center for variation
        cur_y = beam_y_center

        # Optional: make the gap a "pit" for realism (if not final beam)
        if i < 6:  # Don't add pit after final beam
            pit_x1 = x2
            pit_x2 = min(x2 + gap_length_idx, m_to_idx(length)-1)
            height_field[pit_x1:pit_x2, :] = -0.4 - 0.3 * difficulty  # up to 70cm deep

    # Final goal is just off the last beam, where the robot can step down safely
    goals[7] = [min(cur_x + m_to_idx(0.5), m_to_idx(length)-2), mid_y]

    # Make sure the end area is flat so the robot finishes safely
    if cur_x < m_to_idx(length):
        height_field[cur_x:, :] = 0

    return height_field, goals

def set_terrain_5(length, width, field_resolution, difficulty):
    """Wide staircase hurdles: robot must repeatedly step up, walk short on platforms, and step down."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for the course
    hurdle_height_min, hurdle_height_max = 0.07, 0.28   # Low step to nearly hip-high
    hurdle_height = hurdle_height_min + (hurdle_height_max - hurdle_height_min) * difficulty
    platform_length = 0.7 - 0.2 * difficulty           # How long each upstep is walkable
    hurdle_width = 1.2                                # All obstacles at least 1m wide
    step_thickness = 0.07 + 0.04 * difficulty         # Wall thickness
    gap_length = 1.1 - 0.5 * difficulty               # Space between steps (robot walks on ground)
    n_hurdles = 6                                     # Number of steps/platforms

    # Platform sizes (quantized)
    q_hurdle_height = hurdle_height
    q_platform_length = m_to_idx(platform_length)
    q_step_thickness = m_to_idx(step_thickness)
    q_gap_length = m_to_idx(gap_length)
    q_hurdle_width = m_to_idx(hurdle_width)

    mid_y = m_to_idx(width // 2)
    y1 = mid_y - q_hurdle_width // 2
    y2 = mid_y + q_hurdle_width // 2

    # Clear spawning area
    spawn_end = m_to_idx(2)
    height_field[0:spawn_end, :] = 0

    # First goal: just after spawn
    goals[0] = [spawn_end + m_to_idx(0.2), mid_y]

    # Add hurdles
    cur_x = spawn_end
    for h in range(n_hurdles):
        # Wall ("step") - vertical face
        wall_x1 = cur_x
        wall_x2 = cur_x + q_step_thickness
        # Platform on top
        plat_x1 = wall_x2
        plat_x2 = plat_x1 + q_platform_length

        # Raise wall and platform
        height_field[wall_x1:wall_x2, y1:y2] = q_hurdle_height
        height_field[plat_x1:plat_x2, y1:y2] = q_hurdle_height

        # Robot traverses wall and climbs to platform -- so place goal in the center of the flat platform
        goal_x = (plat_x1 + plat_x2) // 2
        goal_y = mid_y
        # Place the goal right on the platform center
        if h < 7:
            goals[h + 1] = [goal_x, goal_y]

        # Next ground section ("descending step" drops straight down)
        next_ground_x1 = plat_x2
        next_ground_x2 = next_ground_x1 + q_gap_length
        height_field[next_ground_x1:next_ground_x2, :] = 0   # Reset to ground

        # Move forward
        cur_x = next_ground_x2

    # Final goal: just after last hurdle
    goals[-1] = [min(cur_x + m_to_idx(0.5), m_to_idx(length)-1), mid_y]

    # Clip everything within the course bounds
    height_field = height_field[:m_to_idx(length), :m_to_idx(width)]

    return height_field, goals

def set_terrain_6(length, width, field_resolution, difficulty):
    """A sequence of seesaw (teeter-totter) bridges: tests balancing on unstable, slanted, narrow surfaces."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters for the seesaw bridges
    seesaw_length = 1.7 - 0.3 * difficulty  # Length in meters
    seesaw_width = 1.1 - 0.3 * difficulty   # Narrows at high difficulty (min 0.5 m)
    seesaw_width = max(seesaw_width, 0.5)
    seesaw_height = 0.11 + 0.14 * difficulty  # Maximum seesaw height diff from pivot point
    gap_length = 0.35 + 0.6 * difficulty     # Gap between seesaws, these are flat ground
    n_seesaws = 5  # Fewer long seesaws rather than many
    mid_y = m_to_idx(width / 2)

    seesaw_length_idx = m_to_idx(seesaw_length)
    seesaw_width_idx = m_to_idx(seesaw_width)
    gap_length_idx = m_to_idx(gap_length)

    # Set flat safe spawn area 
    spawn_length_idx = m_to_idx(2)
    height_field[:spawn_length_idx, :] = 0
    goals[0] = [spawn_length_idx-m_to_idx(0.5), mid_y]  # Start goal

    cur_x = spawn_length_idx
    seesaw_count = 0

    def add_seesaw(x_start, seesaw_len, y_center, seesaw_wid, tilt_sign):
        """Add a sloped seesaw bridge with the given parameters."""
        x_end = x_start + seesaw_len
        y1 = y_center - seesaw_wid//2
        y2 = y_center + seesaw_wid//2
        # Seesaw pivots at midpoint, so first half ramps up, second half ramps down
        for xi in range(x_start, x_end):
            rel = (xi - x_start) / (seesaw_len - 1)
            if rel < 0.5:
                offs = tilt_sign * 2 * seesaw_height * (rel)
            else:
                offs = tilt_sign * 2 * seesaw_height * (1 - rel)
            height_field[xi, y1:y2] = offs
        return (x_start+x_end)//2, (y1+y2)//2

    # Lay out a chain of seesaws along the center y
    for i in range(n_seesaws):
        # Introduce small random y deviation for some lateral challenge
        y_shift = m_to_idx(random.uniform(-0.35, 0.35) * (1-difficulty))
        # Alternate slant direction
        tilt_sign = 1 if i % 2 == 0 else -1
        seesaw_cx, seesaw_cy = add_seesaw(cur_x, seesaw_length_idx, mid_y + y_shift, seesaw_width_idx, tilt_sign)
        goals[i+1] = [seesaw_cx, seesaw_cy]

        cur_x += seesaw_length_idx
        # Insert gap after bridge
        if i < n_seesaws-1:
            height_field[cur_x:cur_x+gap_length_idx, :] = 0
            cur_x += gap_length_idx

    # Fill any remaining space with flat ground
    height_field[cur_x:, :] = 0
    # Set final (8th) goal after all seesaws
    goals[-1] = [min(cur_x + m_to_idx(0.5), height_field.shape[0]-1), mid_y]

    return height_field, goals

def set_terrain_7(length, width, field_resolution, difficulty):
    """Series of tilted balance beams to challenge the robot's dynamic walking and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Balance beam parameters
    # Each beam spans the course width; their heights and slopes increase with difficulty
    beam_length = 1.7 - 0.4 * difficulty   # shorter beams for higher difficulty
    beam_width = 0.42 + 0.25 * (1-difficulty) # slightly wider at low difficulty
    beam_start_height = 0.07 + 0.09 * difficulty
    beam_max_slope = 0.06 + 0.17 * difficulty     # meters elevation over length

    gap_len = 0.27 + 0.48 * difficulty
    num_beams = 6

    total_beam_and_gaps = num_beams * m_to_idx(beam_length) + (num_beams-1) * m_to_idx(gap_len)
    available_length = m_to_idx(length) - m_to_idx(2) - m_to_idx(1) # leave spawn and exit areas

    # Distribute beams and gaps in sequence along x-axis
    x = m_to_idx(2)   # Start after safe spawn zone
    mid_y = m_to_idx(width/2)

    def add_beam(start_x, beam_len, center_y, slope, height_start):
        """Adds a sloped balance beam."""
        beam_width_idx = m_to_idx(beam_width) // 2
        for i in range(m_to_idx(beam_len)):
            h = height_start + slope * (i)
            height_field[start_x+i, center_y-beam_width_idx:center_y+beam_width_idx] = h

    # Set spawn area flat
    height_field[:x, :] = 0
    goals[0] = [x - m_to_idx(1), mid_y] # first goal at the end of spawn area

    for i in range(num_beams):
        # Random slope direction for variety (+ up, - down)
        slope = random.choice([1, -1]) * (beam_max_slope / m_to_idx(beam_length))
        # Random left/right offset (beam center y), small at low difficulty
        beam_offset_range = (0.10 + 0.70 * difficulty) * (width/2 - beam_width/2 - 0.3)
        beam_center_y = mid_y + m_to_idx(random.uniform(-beam_offset_range, beam_offset_range))
        height_start = beam_start_height + random.uniform(-0.03, 0.03)*difficulty
        # Add narrow, sloped beam
        add_beam(x, beam_length, beam_center_y, slope, height_start)
        # Set next goal at middle of beam
        beam_center_x = x + m_to_idx(beam_length/2)
        goals[i+1] = [beam_center_x, beam_center_y]
        # Set gaps between beams to be shallow pit
        pit_x1 = x + m_to_idx(beam_length)
        pit_x2 = pit_x1 + m_to_idx(gap_len)
        if pit_x2 < m_to_idx(length) - m_to_idx(1):
            height_field[pit_x1:pit_x2, :] = -0.22 - 0.15*difficulty
        x = pit_x2

    # Final goal on ground at end of corridor
    if x < m_to_idx(length):
        height_field[x:, :] = 0
        goals[7] = [x + m_to_idx(0.6), mid_y]
    else:
        goals[7] = [m_to_idx(length)-1, mid_y]

    return height_field, goals

def set_terrain_8(length, width, field_resolution, difficulty):
    """U-shaped urban parkour ledge: Run forward, sharp left on balance beam, sharp right on balance beam, jump off."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course layout: U-Shape using urban 'ledges' and 'balancing beams'
    # 1. Run straight, climb up curb/ledge (height depends on difficulty)
    # 2. Sharp left onto a narrow but traversable balance beam
    # 3. Sharp right onto another beam, then jump off

    curb_height = 0.09 + 0.16 * difficulty     # Simulates a curb to climb onto (9-25cm)
    beam_height = curb_height                  # Beams stay at curb height for alignment
    beam_width = 0.32 - 0.10 * difficulty      # Beam gets narrower (32-22cm)
    beam_length = 2.6 + 2.2 * difficulty       # Longer balance at high difficulty (2.6-4.8m)
    landing_height = 0                         # After final jump down
    ledge_length = 1.2                         # Ledge length before beam (1.2m)
    gap_length = 0.17 + 0.20 * difficulty      # Gap that must be jumped across after beam (17-37cm)

    side_margin = 0.34                        # amount of clearance from y-edge, in meters
    start_clear = 2.0                         # flat clear ground at start for spawning

    # Indexing
    x0 = m_to_idx(0)
    x_spawn_end = m_to_idx(start_clear)
    x_ledge_end = x_spawn_end + m_to_idx(ledge_length)
    y_mid = m_to_idx(width / 2)
    y_margin = m_to_idx(side_margin)
    beam_w = max(m_to_idx(beam_width), m_to_idx(0.24)) # never less than 24cm wide

    y_left = y_margin + beam_w // 2
    y_right = m_to_idx(width) - y_margin - beam_w // 2

    # --------- 1. Flat spawn area ---------
    height_field[x0:x_spawn_end, :] = 0
    goals[0] = [m_to_idx(1.0), y_mid]   # initial goal in flat region for heading alignment

    # --------- 2. Forward ledge climb ---------
    height_field[x_spawn_end:x_ledge_end, y_mid - m_to_idx(0.7):y_mid + m_to_idx(0.7)] = curb_height
    goals[1] = [x_spawn_end + m_to_idx(0.6), y_mid]    # center of the ledge

    # --------- 3. 90-degree left: First balance beam ---------
    x_beam_start = x_ledge_end
    x_beam_end = x_beam_start + m_to_idx(beam_length * 0.37)   # 1st leg of beam (turn left)
    y_beam_left_start = y_mid - beam_w // 2
    y_beam_left_end = y_margin + beam_w

    # Beam goes to left edge in y, robot must make about a 90-deg left
    height_field[x_beam_start:x_beam_end, y_beam_left_end - beam_w : y_beam_left_end] = beam_height
    goals[2] = [x_beam_end - m_to_idx(0.18), y_beam_left_end - beam_w // 2]   # near left edge: turn point

    # --------- 4. 90-degree right: Second beam ---------
    # Now move sideways along y at the left, toward the far wall
    y_beam_right_start = y_beam_left_end - beam_w
    y_beam_right_end = y_beam_right_start + m_to_idx(beam_length)
    x_beam2 = x_beam_end                              # same x
    height_field[x_beam2:x_beam2 + beam_w, 
                 y_beam_right_start:y_beam_right_end] = beam_height
    goals[3] = [x_beam2 + beam_w // 2, y_beam_right_end - m_to_idx(0.2)]   # near end of beam: turn point

    # --------- 5. 90-degree right: Third beam forward ---------
    # Head forward on rightmost side
    x_beam3_start = x_beam2 + beam_w
    x_beam3_end = x_beam3_start + m_to_idx(beam_length * 0.42)
    y_beam3 = y_beam_right_end - beam_w
    height_field[x_beam3_start:x_beam3_end, y_beam3:y_beam3 + beam_w] = beam_height
    goals[4] = [x_beam3_end - m_to_idx(0.18), y_beam_right_end - beam_w // 2]

    # --------- 6. Gap to final landing ---------
    x_gap_start = x_beam3_end
    x_gap_end = x_gap_start + m_to_idx(gap_length)
    # No surface on the gap - robot must jump!
    height_field[x_gap_start:x_gap_end, y_beam3:y_beam3 + beam_w] = -0.5

    # Landing zone after final jump
    x_land_start = x_gap_end
    x_land_end = min(m_to_idx(length), x_land_start + m_to_idx(2.5))
    height_field[x_land_start:x_land_end, y_beam3:y_beam3 + m_to_idx(1.0)] = landing_height
    goals[5] = [x_land_start + m_to_idx(0.45), y_beam_right_end - beam_w // 2] # after the jump

    # --------- 7. Optional small drop or step at end ---------
    if difficulty > 0.4:
        final_drop_x0 = x_land_end
        final_drop_x1 = min(m_to_idx(length), final_drop_x0 + m_to_idx(0.65 + 0.35 * difficulty))
        height_field[final_drop_x0:final_drop_x1, y_beam3:y_beam3 + m_to_idx(1.0)] = -0.22
        final_goal_x = final_drop_x1 - m_to_idx(0.1)
        final_goal_y = y_beam_right_end - beam_w // 2
    else:
        final_goal_x = x_land_end - m_to_idx(0.1)
        final_goal_y = y_beam_right_end - beam_w // 2
    goals[6] = [final_goal_x, final_goal_y]

    # --------- Fill in last goal at the course exit ---------
    goals[7] = [m_to_idx(length) - m_to_idx(0.3), y_beam_right_end - beam_w // 2]

    # --------- Clamp bounds to avoid IndexError ---------
    height_field = height_field[:m_to_idx(length), :m_to_idx(width)]
    for i in range(goals.shape[0]):
        goals[i, 0] = np.clip(goals[i, 0], 0, m_to_idx(length)-1)
        goals[i, 1] = np.clip(goals[i, 1], 0, m_to_idx(width)-1)
    
    return height_field, goals

def set_terrain_9(length, width, field_resolution, difficulty):
    """Stepping-stone 'city crossing' with narrow beams, curbs, and angled turns to test precise foot placement and turning."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    #####################
    # Course Parameters #
    #####################

    # Beam parameters (beams as narrow walkways over pits)
    beam_width = 0.45 + 0.3*(1-difficulty)    # Narrows with difficulty, min: 0.45, max: 0.75m
    beam_width_idx = m_to_idx(beam_width)

    beam_height = 0.12 + 0.09*difficulty      # Beam sits above a small pit
    pit_depth = -(0.35 + 0.30*difficulty)     # Pit below = negative field

    # Curb parameters (low, wide steps)
    curb_width = 1.4
    curb_length = 0.7 + 0.5*(1-difficulty)
    curb_width_idx = m_to_idx(curb_width)
    curb_length_idx = m_to_idx(curb_length)
    curb_height = 0.13 + 0.11*difficulty

    mid_y = m_to_idx(width/2)
    total_x = m_to_idx(length)
    total_y = m_to_idx(width)

    # Offset for safe spawning
    spawn_length_idx = m_to_idx(2)
    height_field[:spawn_length_idx, :] = 0
    goals[0] = [m_to_idx(1), mid_y]  # spawn point

    ###############
    # Beams/Pits  #
    ###############
    beam_segments = 2 + int(2 * difficulty)   # 2 beams at easy, up to 4 at hard
    beam_length = (length - 5.0) / (beam_segments+1)  # leave space for curbs at both ends
    beam_length_idx = m_to_idx(beam_length)
    pit_length = 0.48 + 0.55*difficulty
    pit_length_idx = m_to_idx(pit_length)

    curx_idx = spawn_length_idx
    prev_goal = [curx_idx, mid_y]
    goal_num = 1

    for seg in range(beam_segments):
        # Place a pit
        pit_start_x = curx_idx
        pit_end_x = pit_start_x + pit_length_idx
        height_field[pit_start_x:pit_end_x, :] = pit_depth

        # Place beam over pit
        # For challenge, offset the beam side to side at each segment
        if seg % 2 == 0:
            beam_center_y = mid_y - m_to_idx(0.8 + 0.6*difficulty) // 2
        else:
            beam_center_y = mid_y + m_to_idx(0.8 + 0.6*difficulty) // 2

        beam_start_x = pit_start_x
        beam_end_x = pit_end_x
        by1 = beam_center_y - beam_width_idx//2
        by2 = beam_center_y + beam_width_idx//2

        # Ensure within bounds
        by1 = max(by1, m_to_idx(0.1))
        by2 = min(by2, total_y - m_to_idx(0.1))
        height_field[beam_start_x:beam_end_x, by1:by2] = beam_height

        # Place goal at middle of beam
        if goal_num < 8:
            gx = (beam_start_x + beam_end_x)//2
            gy = (by1 + by2)//2
            goals[goal_num] = [gx, gy]
            prev_goal = [gx, gy]
            goal_num += 1

        curx_idx = pit_end_x + m_to_idx(0.18 + 0.1*difficulty)  # Slight gap for flat ground

        # Flat area for stability on landing
        height_field[curx_idx:curx_idx+m_to_idx(0.45), :] = 0

        curx_idx += m_to_idx(0.45)

    ##############
    # Curb Steps #
    ##############
    # Place final curb series requiring single or double step-up, maybe with turns

    curb_count = 2 if difficulty < 0.5 else 3
    turn_angle = 0.45 + 0.5*difficulty  # Amount of "turn" at each curb step
    curb_start_y = mid_y

    for curb in range(curb_count):
        cl = curb_length_idx
        cw = curb_width_idx
        ch = curb_height

        curb_x1 = curx_idx
        curb_x2 = curb_x1 + cl

        # Place curb step at angle: left, right, center
        direction = -1 if curb % 2 == 0 else 1
        offset = direction * m_to_idx(turn_angle * (curb+1) * (1-difficulty+0.5))
        curb_y1 = curb_start_y + offset - cw//2
        curb_y2 = curb_start_y + offset + cw//2
        curb_y1 = max(m_to_idx(0.1), min(total_y - cw - m_to_idx(0.1), curb_y1))
        curb_y2 = curb_y1 + cw

        # Raise the platform for curb
        height_field[curb_x1:curb_x2, curb_y1:curb_y2] = ch

        # Place goal at center of curb step
        if goal_num < 8:
            gx = (curb_x1 + curb_x2)//2
            gy = (curb_y1 + curb_y2)//2
            goals[goal_num] = [gx, gy]
            prev_goal = [gx, gy]
            goal_num += 1

        # Move curx_idx for next curb
        curx_idx = curb_x2

    # Final goal after last curb
    if goal_num < 8:
        final_goal_x = min(curx_idx + m_to_idx(0.9), height_field.shape[0]-1)
        final_goal_y = prev_goal[1]
        goals[goal_num] = [final_goal_x, final_goal_y]

    # Fill unused goals with last goal position (so the array is always size 8)
    for i in range(goal_num+1, 8):
        goals[i] = goals[goal_num]

    # Ensure entire area is padded below 0 if necessary
    # Make sure spawn and finish zones are always flat
    height_field[0:spawn_length_idx, :] = 0
    height_field[-m_to_idx(1):, :] = 0

    return height_field, goals

# INSERT TERRAIN FUNCTION DEFINITIONS HERE
