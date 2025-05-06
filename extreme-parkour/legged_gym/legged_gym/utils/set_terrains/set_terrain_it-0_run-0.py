
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
    """Alternating angled ramps for testing balancing and ascending/descending skills."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return (np.round(m / field_resolution).astype(np.int16)
                if not (isinstance(m, list) or isinstance(m, tuple))
                else [round(i / field_resolution) for i in m])

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course parameters
    total_ramps = 6
    spawn_length = m_to_idx(2)
    length_avail = m_to_idx(length) - spawn_length
    mid_y = m_to_idx(width // 2)
    course_width = m_to_idx(width)
    safe_margin = m_to_idx(0.5)  # margin on the sides for safety

    # Ramp geometry
    ramp_min_length = 1.2      # meters
    ramp_max_length = 2.0
    ramp_length = np.linspace(
        ramp_min_length, ramp_max_length - difficulty * 0.5, total_ramps)
    ramp_length = [m_to_idx(l) for l in ramp_length]
    flat_top_length = m_to_idx(0.3 + 0.2 * (1 - difficulty))

    ramp_min_width = 1.2      # meters, always >1.0 as per spec
    ramp_max_width = min(width - 0.5, 2.0)
    ramp_w = np.linspace(ramp_min_width, ramp_max_width, total_ramps)
    ramp_widths = [m_to_idx(w) for w in ramp_w]

    # Ramp heights
    max_height = 0.25 + 0.15 * difficulty
    min_height = 0.04 + 0.1 * difficulty  # don't make too easy

    # -1 for descent, 1 for ascent
    ramp_directions = [1 if i % 2 == 0 else -1 for i in range(total_ramps)]
    # Each ramp is at a random orientation: sometimes angled forward, sometimes left/right, so need to keep the centerline zig-zag
    y_offsets = np.linspace(safe_margin, course_width - safe_margin, total_ramps + 2)[1:-1]
    y_offsets = [int(o) for o in y_offsets]

    cur_x = spawn_length

    # Set spawn area to flat ground.
    height_field[:spawn_length, :] = 0
    goals[0] = [spawn_length - m_to_idx(0.5), mid_y]

    for i in range(total_ramps):
        # Ramps alternate in ascent/descent
        direction = ramp_directions[i]
        length = ramp_length[i]
        width_ = ramp_widths[i]

        # Height from base to top of ramp
        ramp_height = np.random.uniform(min_height, max_height) * direction

        # Center the ramp along the y-axis, but shift to zig-zag y location.
        ramp_mid_y = y_offsets[i]

        x0 = cur_x
        x1 = cur_x + length
        y0 = max(ramp_mid_y - width_ // 2, 0)
        y1 = min(ramp_mid_y + width_ // 2, course_width)

        # Ramp slope (linear interpolation between end points)
        for xi in range(x0, x1):
            frac = (xi - x0) / max(1, x1 - x0 - 1)
            h = ramp_height * frac
            height_field[xi, y0:y1] = (
                height_field[xi, y0:y1] + h
            )  # relative to starting ground (which might be nonzero)

        # Flat platform on top of ramp for stability
        xf0, xf1 = x1, min(x1 + flat_top_length, m_to_idx(length))
        top_height = ramp_height
        height_field[xf0:xf1, y0:y1] = height_field[x1 - 1, y0:y1]
        cur_x = xf1

        # Intermediate goal: always place on the middle "flat" top of ramp
        # Or if i == last, place further along the flat ground
        if i < 7:
            goal_y = int((y0 + y1) // 2)
            goal_x = int((x1 + xf0) // 2)
            goals[i + 1] = [goal_x, goal_y]

    # Final ramp-off platform and goal
    end_pad = m_to_idx(1)
    height_field[cur_x:cur_x+end_pad, :] = 0  # return to ground level
    goals[7] = [min(cur_x + end_pad // 2, m_to_idx(length) - 2), mid_y]

    # Clip all heights so the minimum is 0
    min_spawn_height = height_field[:spawn_length, :].min()
    if min_spawn_height > 0:
        height_field -= min_spawn_height

    # Ensure all goals are within bounds
    for i in range(8):
        goals[i, 0] = min(max(0, goals[i, 0]), m_to_idx(length) - 1)
        goals[i, 1] = min(max(0, goals[i, 1]), m_to_idx(width) - 1)

    return height_field, goals

def set_terrain_1(length, width, field_resolution, difficulty):
    """Stepping stone path: Repeated narrow raised pads and gaps for precise foot placement and jumping."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- Stepping Stone Parameters ---
    #
    # The stepping stone pads are like large paving stones: flat, slightly raised pads with gaps between them.
    # The pads get a little smaller and the gaps a little bigger with increased difficulty.
    #
    # All pads extend across most of the width of the course, but with random lateral offsets to force steering.
    # The quadruped must step onto these single-passage pads (avoiding the 'water' -- negative height gaps).
    #
    # At high difficulty, stones are just wide enough for the robot to land with little tolerance.
    #

    # Parameters in meters
    pad_length = 0.7 - 0.15 * difficulty    # Stepping stone (along x)
    pad_width_min = 0.7 - 0.25 * difficulty # Narrowness (along y)
    pad_width_max = 1.2 - 0.35 * difficulty
    gap_length = 0.37 + 0.43 * difficulty   # Distance between stones (x)
    gap_depth = -0.16 - 0.24 * difficulty   # Height of negative space between stones

    # For lateral displacement
    lateral_max_offset = 1.0 - 0.75 * difficulty  # Max offset reduces on hard (forces more straight line at hardest)
    pad_overlap = 0.18 + (0.1 * (1-difficulty))   # How much "lead-on" the pad gives in y to allow steering transitions

    # Spawn and goal logic
    pad_count = 6
    spawn_length = 2.0   # meters before first pad must not have obstacles
    cur_x = m_to_idx(spawn_length)

    # Place initial area (the spawn zone) as plain ground
    height_field[:cur_x, :] = 0
    y_center_idx = m_to_idx(width/2)

    # Place first goal at spawn position
    goals[0] = [m_to_idx(1.0), y_center_idx]

    # Helper function to add a pad at (start_x, y_center)
    def add_pad(start_x, y_center, pad_len_idx, pad_wid_idx, stone_height):
        y1 = max(0, y_center - pad_wid_idx//2)
        y2 = min(height_field.shape[1], y_center + pad_wid_idx//2)
        x1 = start_x
        x2 = min(height_field.shape[0], start_x + pad_len_idx)
        height_field[x1:x2, y1:y2] = stone_height

    # All gaps between stones are negative
    height_field[cur_x:, :] = gap_depth

    stone_height = 0.06 + 0.18*difficulty  # A little step up at easy, moderate at hard

    lateral_centers = []
    for i in range(pad_count):
        # Keep pads well in bounds, avoiding too close to edges
        pad_width = random.uniform(pad_width_min, pad_width_max)
        pad_wid_idx = m_to_idx(pad_width)
        pad_len_idx = m_to_idx(pad_length)

        # Pick center y coordinate randomly, avoiding too close to left/right side
        left_bound = m_to_idx(0.7) + pad_wid_idx//2
        right_bound = height_field.shape[1] - m_to_idx(0.7) - pad_wid_idx//2
        # Allow up to lateral_max_offset in y
        if i == 0:
            y_center = y_center_idx  # Start straight
        else:
            max_offset = m_to_idx(lateral_max_offset)
            prev_center = lateral_centers[-1]
            # Pick new center within offset limits, and course bounds
            y_center = np.clip(prev_center + random.randint(-max_offset, max_offset), left_bound, right_bound)
        lateral_centers.append(y_center)

        # Place pad
        add_pad(cur_x, y_center, pad_len_idx, pad_wid_idx, stone_height)

        # Place goal in the middle of this pad
        pad_center_x = cur_x + pad_len_idx//2
        goals[i+1] = [pad_center_x, y_center]

        # Move to next stone, with randomized gap for variety
        gap = gap_length + random.uniform(-0.05, 0.07)*difficulty
        cur_x += pad_len_idx + m_to_idx(gap)

    # Final pad ("dry land") to finish
    end_pad_x = min(cur_x, height_field.shape[0] - m_to_idx(0.9))
    final_pad_len = m_to_idx(1.1)
    final_pad_width = m_to_idx(1.3)
    add_pad(end_pad_x, y_center_idx, final_pad_len, final_pad_width, stone_height)
    goals[7] = [end_pad_x + final_pad_len//2, y_center_idx]

    # Ensure the rest of the course after last pad is regular ground (not negative)
    height_field[end_pad_x+final_pad_len:, :] = 0

    return height_field, goals

def set_terrain_2(length, width, field_resolution, difficulty):
    """Alternating narrow balance beams and wide low platforms, testing balance control and precise foot placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Constants and robot size reference (0.645x0.28 meters)
    course_len_idx = m_to_idx(length)
    course_wid_idx = m_to_idx(width)
    spawn_x = m_to_idx(1)
    mid_y = course_wid_idx // 2
    num_beams = 4  # Odd goals at start/end of beams
    num_platforms = 4  # Even goals at end of platforms
    beam_length = 1.4 + difficulty * 1.3  # Beams get longer with difficulty
    beam_length_idx = m_to_idx(beam_length)
    beam_width = 0.45 + 0.15 * (1-difficulty)  # 0.6m at easy, 0.45m at hard (still safe)
    beam_width_idx = m_to_idx(beam_width)

    beam_height = 0.08 + 0.07 * difficulty  # Beams slightly above ground, up to 0.15m
    platform_length = 1.0 + difficulty * 0.7
    platform_length_idx = m_to_idx(platform_length)
    platform_width = 1.4 + 0.6 * (1-difficulty)  # Platforms wider at easy, min 1.4m
    platform_width_idx = m_to_idx(platform_width)
    platform_height = 0.05 + 0.02 * (random.random() - 0.5)  # Platforms flush or slightly above ground

    gap_length = 0.13 + difficulty * 0.22  # 0.13~0.35m gaps between obstacles
    gap_length_idx = m_to_idx(gap_length)

    # Set spawn area: at least 2m, clear of obstacles
    spawn_clear = m_to_idx(2.0)
    height_field[:spawn_clear, :] = 0
    goals[0] = [spawn_x, mid_y]  # First goal is right past spawn

    # Start building course, alternating beams and platforms, placing goals at obstacle ends
    cur_x = spawn_clear
    y_offset_choices = [0, m_to_idx(0.35), -m_to_idx(0.35)]  # Sometimes beams/platforms slightly laterally offset

    for i in range(1, 8):
        if i % 2 == 1:
            # Beam
            x1 = cur_x
            x2 = cur_x + beam_length_idx
            # Slightly vary y for challenge, prevent going out of bounds
            y_center = mid_y + random.choice(y_offset_choices)
            y1 = max(0, y_center - beam_width_idx // 2)
            y2 = min(course_wid_idx, y_center + int(np.ceil(beam_width_idx / 2)))
            # The rest of the ground at this x is a pit
            height_field[x1:x2, :] = -0.35 - 0.2 * difficulty
            # Draw the beam above the pit
            height_field[x1:x2, y1:y2] = beam_height
            # Set goal at the end of the beam
            goals[i] = [x2 - m_to_idx(0.18), (y1 + y2)//2]
            cur_x = x2 + gap_length_idx
        else:
            # Wide, safe platform
            x1 = cur_x
            x2 = cur_x + platform_length_idx
            y_center = mid_y + random.choice(y_offset_choices)
            y1 = max(0, y_center - platform_width_idx // 2)
            y2 = min(course_wid_idx, y_center + int(np.ceil(platform_width_idx / 2)))
            height_field[x1:x2, :] = 0  # Reset pit
            height_field[x1:x2, y1:y2] = platform_height
            # Set goal at the center of the platform
            goals[i] = [x2 - m_to_idx(0.20), (y1 + y2)//2]
            cur_x = x2 + gap_length_idx

    # Clear the final section
    height_field[cur_x:, :] = 0
    # Ensure last goal is in reach
    goals[7] = [min(cur_x + m_to_idx(0.3), course_len_idx - m_to_idx(0.3)), mid_y]

    # Guarantee all goals are within bounds and not inside pits
    for i in range(8):
        goals[i, 0] = np.clip(goals[i, 0], 0, course_len_idx-1)
        goals[i, 1] = np.clip(goals[i, 1], 0, course_wid_idx-1)

    return height_field, goals

def set_terrain_3(length, width, field_resolution, difficulty):
    """Slalom course with alternating wide step-over barriers that force zig-zag lateral turns."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters
    spawn_x = m_to_idx(1.0)
    mid_y = m_to_idx(width / 2)
    n_barriers = 6  # 6 alternating barriers to generate 7 weaving segments
    barrier_width = m_to_idx(1.00)  # the barriers run the full width, but the path is cut out
    barrier_thickness = m_to_idx(0.20)  # thickness along the x-axis
    min_channel_w = 1.1 - 0.6*difficulty  # meters, 1.1m (easy) to 0.5m (hard)
    min_channel_w = max(0.5, min_channel_w)
    channel_width = m_to_idx(min_channel_w)
    barrier_height = 0.14 + 0.21*difficulty  # from 0.14m (easy) to 0.35m (hard)
    
    # Compute barrier positions
    margin_x = m_to_idx(1.0)  # Give at least 1 m before first barrier after spawn
    start_x = spawn_x + margin_x
    dx = (m_to_idx(length) - start_x - m_to_idx(1.0)) // (n_barriers+1)
    x_positions = [int(start_x + (i+1)*dx) for i in range(n_barriers)]

    # Y-positions for open channels (alternate sides)
    # On each barrier, leave a channel at either hard left or hard right.
    edge_padding = m_to_idx(0.15)
    left_channel_center  = edge_padding + channel_width//2
    right_channel_center = m_to_idx(width) - edge_padding - channel_width//2

    # Sides alternate for each barrier: left, right, left, right...
    channel_centers = [left_channel_center if i%2==0 else right_channel_center for i in range(n_barriers)]

    # Spawn/initial goal
    goals[0] = [spawn_x//2, mid_y]

    prev_goal_x = spawn_x
    prev_goal_y = mid_y

    for i, (x_b, y_c) in enumerate(zip(x_positions, channel_centers)):
        # Place barrier
        y_start = 0
        y_end = m_to_idx(width)
        x1 = x_b
        x2 = min(x1 + barrier_thickness, m_to_idx(length))

        # Clear channel in barrier at correct side
        c_center = int(y_c)
        c_half = channel_width // 2

        # Set the whole barrier
        height_field[x1:x2, y_start:y_end] = barrier_height

        # Cut out a channel
        y1 = max(int(c_center - c_half), 0)
        y2 = min(int(c_center + c_half), m_to_idx(width))
        height_field[x1:x2, y1:y2] = 0.0  # clear out the path

        # Set goal just past the channel to force zig-zag
        # Place it halfway beyond the barrier and centered in the open channel
        next_goal_x = min(x2 + m_to_idx(0.30), m_to_idx(length)-1)  # move 0.3m forward from barrier
        next_goal_y = c_center
        goals[i+1] = [next_goal_x, next_goal_y]

        prev_goal_x = next_goal_x
        prev_goal_y = next_goal_y

    # Final straight goal at the end:
    final_goal_x = m_to_idx(length) - m_to_idx(0.50)
    # Final goal alternates again
    final_goal_y = left_channel_center if n_barriers%2==0 else right_channel_center
    goals[-1] = [final_goal_x, final_goal_y]

    # Keep the final bit flat to let the robot finish
    height_field[final_goal_x:, :] = 0

    return height_field, goals

def set_terrain_4(length, width, field_resolution, difficulty):
    """
    Stepping Stone Balance Course: Series of narrow, staggered raised rectangular stepping stones across a shallow water pit,
    forcing the quadruped to carefully step and balance as it traverses, emphasizing precise foot placement and lateral movement.
    The obstacles are slightly offset laterally and spaced so the robot must weave left/right between goals.
    """

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Terrain parameters
    course_len_idx = m_to_idx(length)
    course_wid_idx = m_to_idx(width)
    spawn_idx_x = m_to_idx(1)
    min_pad = m_to_idx(0.2)

    # Parameters for stepping stones (platforms)
    stone_length = 0.6 + 0.4 * (1 - difficulty)    # [0.6, 1] meters: harder->shorter
    stone_length_idx = m_to_idx(stone_length)
    stone_width = 0.5 + 0.15 * (1 - difficulty)    # [0.5, 0.65] meters: narrow but always sufficient
    stone_width_idx = m_to_idx(stone_width)
    edge_clearance = m_to_idx(0.20)

    pit_depth = -(0.10 + 0.20 * difficulty)   # up to 0.3m at max difficulty, shallow at low difficulty
    stone_height = 0.00 + 0.04 * difficulty   # near ground at easy, up to 4cm above ("floating") at hard

    lateral_offset_range = m_to_idx(0.7) if difficulty > 0.5 else m_to_idx(0.4)
    inter_stone_gap = 0.5 + 0.55 * difficulty   # [0.5, 1.05] meters
    inter_stone_gap_idx = m_to_idx(inter_stone_gap)

    # 1. Set "pit": the central strip of the arena is a water pit, the spawn and ending segments are on solid ground
    height_field[:, :] = 0.     # default
    # Backfill the "water pit"
    pit_start = spawn_idx_x + m_to_idx(0.5)
    pit_end = m_to_idx(length) - m_to_idx(0.5)
    height_field[pit_start:pit_end, :] = pit_depth

    # 2. Set the first and last areas as solid ground
    height_field[0:pit_start, :] = 0
    height_field[pit_end:, :] = 0

    # 3. Place stepping stones in staggered fashion
    # Place the first stone at 1.5 m from start, and then each ~1.0-1.6 m further, alternating their lateral offset
    # Start near the center widthwise, and alternate left-right
    num_stones = 6  # with 1 start, 1 end goal

    stone_xs = [spawn_idx_x + m_to_idx(0.3)]
    gap_variance = m_to_idx(0.15 + 0.3 * difficulty)  # more gap randomness at higher difficulty
    for i in range(1, num_stones):
        prev_x = stone_xs[-1]
        gap = inter_stone_gap_idx + random.randint(-gap_variance, gap_variance)
        new_x = prev_x + stone_length_idx + gap
        if new_x+stone_length_idx >= m_to_idx(length) - edge_clearance:
            break
        stone_xs.append(new_x)

    stone_ys = []
    # Create a gentle stagger: +- lateral_offset within bounds
    center_y = course_wid_idx // 2
    staggering_dir = 1
    for i in range(len(stone_xs)):
        lateral_jitter = (random.randint(-m_to_idx(0.08), m_to_idx(0.08)))  # small randomness for realism
        offset = staggering_dir * (random.randint(int(0.6*lateral_offset_range), lateral_offset_range))
        stone_y = np.clip(center_y + offset + lateral_jitter, edge_clearance, course_wid_idx-edge_clearance)
        stone_ys.append(stone_y)
        staggering_dir *= -1  # alternate left/right

    # 4. Place stones on the height field
    for i, (stone_x, stone_y) in enumerate(zip(stone_xs, stone_ys)):
        x1 = int(max(stone_x - stone_length_idx // 2, pit_start + min_pad))
        x2 = int(min(stone_x + stone_length_idx // 2, pit_end - min_pad))
        y1 = int(max(stone_y - stone_width_idx // 2, edge_clearance))
        y2 = int(min(stone_y + stone_width_idx // 2, course_wid_idx-edge_clearance))
        height_field[x1:x2, y1:y2] = stone_height  # stones rise just above pit
        # Place goal at stone center
        goals[i+1] = [0.5*(x1+x2), 0.5*(y1+y2)]

    # 5. Set precise start and final goals
    # Start goal at spawn on solid ground, centered
    goals[0] = [m_to_idx(0.8), center_y]
    # Final goal is after last stone, on solid ground
    end_x = int(min(stone_xs[-1] + stone_length_idx + 2*inter_stone_gap_idx, course_len_idx-m_to_idx(0.3)))
    goals[-1] = [end_x, center_y + random.randint(-m_to_idx(0.12), m_to_idx(0.12))]

    # 6. Fill extra goals if <8 with linear interpolation along the path
    # This ensures all goals are valid
    idx_of_last = np.count_nonzero(np.any(goals != 0, axis=1)) - 1
    if idx_of_last < 7:
        for j in range(idx_of_last+1, 8):
            v = (j-idx_of_last)/(8-idx_of_last)
            goals[j] = (1-v)*goals[idx_of_last] + v*goals[-1]

    return height_field, goals

def set_terrain_5(length, width, field_resolution, difficulty):
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

def set_terrain_6(length, width, field_resolution, difficulty):
    """A sequence of sloped ramps (uphill and downhill) alternating direction, testing the quadruped's ability to ascend, descend, and change heading while maintaining balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))
    
    # COURSE CONFIGURATION ---------------------------
    # We will alternate sloped ramps going up and down
    # Each ramp will span ~1.4m to 1.8m length, width 1.3 to 1.8m
    # Slope angle and ramp height scale with difficulty

    min_ramp_len, max_ramp_len = 1.4, 1.8
    min_ramp_wid, max_ramp_wid = 1.3, 1.8
    max_total_height = 0.20 + 0.35 * difficulty  # Steeper slope at higher difficulty

    # We'll use 6 ramps, each with an up or down depending on position, alternating direction
    # Space between ramps is flat
    n_ramps = 6
    flat_space_len = 0.40 + 0.12 * (1 - difficulty) # shorter rest space at higher difficulty

    # The course will zig-zag: first ramp up, then turn left, ramp down, then turn right, etc.
    # We'll select y positions to zig-zag across the width
    width_margin = 0.45  # leave some gap on sides for safe navigation
    ramp_centers_y = np.linspace(width_margin, width - width_margin, n_ramps + 1)
    # Random swap every other direction
    if random.random() > 0.5:
        ramp_centers_y = ramp_centers_y[::-1]

    # Place the spawn and first goal at flat area at (1m, center)
    spawn_x = 1.0
    spawn_x_idx = m_to_idx(spawn_x)
    goals[0] = [spawn_x_idx, m_to_idx(width/2)]

    # Set spawn area to flat
    height_field[:m_to_idx(2.0), :] = 0.0

    cur_x = 2.0  # start placing obstacles after safe spawn

    ramp_goals_idx = 1
    for ramp_i in range(n_ramps):
        # Ramp parameters
        ramp_length_m = np.random.uniform(min_ramp_len, max_ramp_len)
        ramp_width_m = np.random.uniform(min_ramp_wid, max_ramp_wid)
        ramp_height = (max_total_height * ((-1) ** ramp_i))  # alternate up and down ramps
        ramp_start_x = cur_x
        ramp_end_x = cur_x + ramp_length_m
        ramp_mid_y = ramp_centers_y[ramp_i % (n_ramps + 1)]

        start_y = np.clip(m_to_idx(ramp_mid_y - ramp_width_m/2), 0, m_to_idx(width)-1)
        end_y = np.clip(m_to_idx(ramp_mid_y + ramp_width_m/2), 1, m_to_idx(width))
        y_slice = slice(start_y, end_y)
        x_start_idx = m_to_idx(ramp_start_x)
        x_end_idx = m_to_idx(ramp_end_x)
        x_slice = slice(x_start_idx, x_end_idx)

        # Compute slope for every x across the ramp
        ramp_len_idx = x_end_idx - x_start_idx
        ramp_slope = np.linspace(0, ramp_height, ramp_len_idx).reshape(-1, 1)
        # Broadcast slope over width
        height_field[x_slice, y_slice] = height_field[x_slice, y_slice] + ramp_slope

        # Set sides flat to force following the ramp
        if start_y > 0:
            height_field[x_slice, :start_y] = -0.15  # 15cm pit
        if end_y < m_to_idx(width):
            height_field[x_slice, end_y:] = -0.15

        # Intermediate flat section after each ramp for stability and goal
        flat_length_m = flat_space_len
        flat_start_x = ramp_end_x
        flat_end_x = ramp_end_x + flat_length_m
        x_flat_start_idx = m_to_idx(flat_start_x)
        x_flat_end_idx = m_to_idx(flat_end_x)
        # Maintain ending height of ramp throughout the flat
        landing_height = ramp_height
        if ramp_i == 0:
            # Ramp height is relative to previous ground, which may not be zero
            base_height = 0.0
        else:
            base_height = height_field[m_to_idx(cur_x-0.01), m_to_idx(ramp_mid_y)]
        base_height = height_field[x_start_idx, m_to_idx(ramp_mid_y)]
        height_field[x_flat_start_idx:x_flat_end_idx, y_slice] = base_height + ramp_height

        # Put a goal at the end of the ramp
        y_goal = np.clip(m_to_idx(ramp_mid_y), 0, m_to_idx(width)-1)
        x_goal = int((x_end_idx + x_flat_start_idx)//2)
        goals[ramp_goals_idx] = [x_goal, y_goal]
        ramp_goals_idx += 1

        cur_x = flat_end_x

    # Pad the rest of the field flat
    height_field[m_to_idx(cur_x):, :] = height_field[m_to_idx(cur_x)-1, :][np.newaxis, :]

    # If less than 8 goals, place final ones at the end, spaced out
    last_goal_x = m_to_idx(cur_x) + m_to_idx(0.2)
    for i in range(ramp_goals_idx, 8):
        goals[i] = [min(last_goal_x + m_to_idx(0.3*(i - ramp_goals_idx)), m_to_idx(length)-2), m_to_idx(width/2)]

    return height_field, goals

def set_terrain_7(length, width, field_resolution, difficulty):
    """Alternating low steps and low balance beams: tests precise foot placement and stable walking across narrow paths."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Parameters based on quadruped size and environment specs
    # All beams and steps have width >= 1m unless otherwise specified as balance beam (narrow, but > 0.4m)
    step_length = 0.8 + 0.3 * difficulty     # step (ledge) the quadruped must climb onto
    step_width = 1.4                        # wide enough for robust stepping
    step_height_min, step_height_max = 0.09, 0.23
    step_height = step_height_min + (step_height_max - step_height_min) * difficulty

    beam_length = 1.7 + 0.3 * difficulty     # long balance beam walk
    beam_width = 0.42 + 0.11 * (1-difficulty)  # balance beam: between 0.42m and 0.53m
    beam_height_min, beam_height_max = 0.12, 0.25
    beam_height = beam_height_min + (beam_height_max - beam_height_min) * difficulty

    pit_depth = -0.9 - 0.2 * difficulty      # deep enough to prevent stepping down, forces using obstacles
    spawn_length = m_to_idx(2)
    mid_y = m_to_idx(width) // 2

    # Ensure narrow terrain features still "fit" the robot (minimum width for beams = 0.42m)
    def add_step(x_start, x_end, y_center, height):
        y1 = y_center - m_to_idx(step_width) // 2
        y2 = y_center + m_to_idx(step_width) // 2
        height_field[x_start:x_end, y1:y2] = height

    def add_beam(x_start, x_end, y_center, height):
        y1 = y_center - m_to_idx(beam_width) // 2
        y2 = y_center + m_to_idx(beam_width) // 2
        height_field[x_start:x_end, y1:y2] = height

    # Fill spawn area with flat ground at height 0
    height_field[:spawn_length, :] = 0
    # Place first goal in the center of starting platform
    goals[0] = [m_to_idx(1), mid_y]

    cur_x = spawn_length
    step_size = m_to_idx(step_length)
    beam_size = m_to_idx(beam_length)
    # For each obstacle segment, alternate step and beam, for a total of 3 each, with safe ground at the end
    for i in range(3):
        # Step (ledge) 
        add_step(cur_x, cur_x + step_size, mid_y, step_height)
        goals[2 * i + 1] = [(cur_x + cur_x + step_size) // 2, mid_y]

        cur_x += step_size
        gap = m_to_idx(0.25 + 0.2 * difficulty)
        cur_x += gap  # Pit between step and beam
        height_field[cur_x - gap:cur_x, :] = pit_depth

        # Balance beam
        side_offset = 0
        # Optional: vary beam center left-right for more variety as difficulty increases
        if difficulty > 0.35:
            side_offset = int(m_to_idx((random.random() - 0.5) * 0.5 * difficulty)) # up to 0.25m side offset
        add_beam(cur_x, cur_x + beam_size, mid_y + side_offset, beam_height)
        goals[2 * i + 2] = [(cur_x + cur_x + beam_size) // 2, mid_y + side_offset]

        cur_x += beam_size
        gap = m_to_idx(0.21 + 0.13 * difficulty)
        cur_x += gap  # Pit between beam and next step
        height_field[cur_x - gap:cur_x, :] = pit_depth

    # Final platform goal: broad, safe exit area
    exit_start = cur_x
    exit_end = m_to_idx(length)
    height_field[exit_start:exit_end, :] = 0
    goals[7] = [exit_start + (exit_end - exit_start) // 2, mid_y]

    return height_field, goals

def set_terrain_8(length, width, field_resolution, difficulty):
    """A sequence of alternating low hurdles and narrow balance beams testing stepping and stability."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Course parameters
    # The quadruped is ~0.645m long, 0.28m wide. Obstacles are sized with respect to this.
    mid_y = m_to_idx(width) // 2

    # Obstacle pattern: hurdle -> beam -> hurdle -> beam ... etc.
    num_obstacles = 6
    safe_margin_x = m_to_idx(2.0)  # Spawn area
    course_x_max = m_to_idx(length)
    course_y_max = m_to_idx(width)

    # Hurdle parameters
    hurdle_width = m_to_idx(1.2)    # Always at least 1m wide
    # Hurdle height increases with difficulty
    hurdle_height = 0.06 + 0.16 * difficulty # 6cm at easy, up to 22cm at hard
    hurdle_length = m_to_idx(0.25 + 0.1 * difficulty)  # Slightly thicker at high diff

    # Beam parameters (balance beam)
    beam_length = m_to_idx(1.1 + 0.4 * difficulty)      # Beams get longer as it gets harder
    # Beam width is challenging but not impossible (0.3 at easy, 0.18 at hard)
    beam_width = m_to_idx(0.3 - 0.12 * difficulty)
    beam_height = 0.10 + 0.10 * difficulty              # 10cm at easy, up to 20cm at hard

    # Distances between obstacles
    gap_between = m_to_idx(0.6 + 0.5 * (1-difficulty))  # Smaller gaps at harder difficulties

    # (x, y) placement tracking
    x = safe_margin_x
    last_y = mid_y

    # Set spawn region flat and goal in center
    height_field[:safe_margin_x, :] = 0
    goals[0] = [safe_margin_x // 2, last_y]

    obstacle_indices = []

    for obs in range(num_obstacles):
        # Alternate between hurdle and beam, place a goal after each obstacle
        if obs % 2 == 0:
            # Hurdle: spans basically the entire width, with one random offset
            center_y = random.randint(m_to_idx(0.7), course_y_max - m_to_idx(0.7))
            y1 = max(0, center_y - hurdle_width // 2)
            y2 = min(course_y_max, center_y + hurdle_width // 2)
            x1 = min(x, course_x_max - hurdle_length)
            x2 = min(x + hurdle_length, course_x_max)

            height_field[x1:x2, y1:y2] = hurdle_height

            # Place goal some distance after obstacle (flat ground)
            goal_x = min(x2 + (gap_between // 2), course_x_max - 1)
            goals[obs + 1] = [goal_x, center_y]
            obstacle_indices.append(('hurdle', (x1, x2, y1, y2)))
            x = x2 + gap_between

            last_y = center_y

        else:
            # Balance beam: position the beam randomly left/right but within central corridor
            beam_center_y = random.randint(m_to_idx(1.1), course_y_max - m_to_idx(1.1))
            half_beam = beam_width // 2
            y1 = max(0, beam_center_y - half_beam)
            y2 = min(course_y_max, beam_center_y + half_beam)
            x1 = min(x, course_x_max - beam_length)
            x2 = min(x + beam_length, course_x_max)

            height_field[x1:x2, y1:y2] = beam_height

            # Place the goal at the far end of the beam (centered on beam)
            goal_x = min(x2 - 1, course_x_max - 1)
            goals[obs + 1] = [goal_x, beam_center_y]
            obstacle_indices.append(('beam', (x1, x2, y1, y2)))
            x = x2 + gap_between

            last_y = beam_center_y

    # Final goal at remaining flat area, at center y
    last_goal_x = min(course_x_max - m_to_idx(1.0), x)
    goals[-1] = [last_goal_x, last_y]
    height_field[x:, :] = 0  # Ensure final part is flat

    # (OPTIONAL: Pit for failed beam crossing? Skipped to emphasize balance, not jumps.)

    # Quick check: Make sure all goal indices are within field bounds
    goals = np.clip(goals, 0, np.array(height_field.shape) - 1)

    return height_field, goals

def set_terrain_9(length, width, field_resolution, difficulty):
    """A sequence of alternating low hurdles (bars) and short stairs, testing stepping and climbing precision."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Configuration parameters (all scale with difficulty)
    mid_y = m_to_idx(width / 2)
    spawn_length = m_to_idx(2)
    height_field[:spawn_length, :] = 0.0  # Flat spawn area

    # Hurdle (bar) parameters:
    hurdle_height_min = 0.07 + 0.03 * difficulty     # lowest bar step at lowest diff (7cm), 10cm at hardest
    hurdle_height_max = 0.13 + 0.10 * difficulty     # up to 23cm at hardest
    hurdle_width = m_to_idx(1.2)                     # at least 1m wide

    # Stair parameters:
    stair_steps_min = 2
    stair_steps_max = 4
    stair_depth_per_step = 0.30                      # meters (enough for one stride)
    stair_rise_per_step = 0.07 + 0.06 * difficulty   # step height, 7-13cm per step

    n_obstacles = 6

    # Inter-obstacle gaps
    gap_base = 0.60 + 0.20 * difficulty    # horizontal ground between obstacles (in meters, scales with diff)
    gap_base_idx = m_to_idx(gap_base)
    bar_len = m_to_idx(0.08)               # thickness of bar/hurdle

    # The sequence: hurdle -> stairs -> hurdle -> stairs ...
    cur_x = spawn_length
    for i in range(n_obstacles):
        center_y = mid_y + random.randint(-m_to_idx(0.15), m_to_idx(0.15))

        # Place hurdle/bar
        if i % 2 == 0:
            # Randomize height for variety
            bar_height = np.random.uniform(hurdle_height_min, hurdle_height_max)
            # Hurdle position
            bar_x1 = cur_x
            bar_x2 = bar_x1 + bar_len
            bar_y1 = center_y - hurdle_width // 2
            bar_y2 = center_y + hurdle_width // 2
            height_field[bar_x1:bar_x2, bar_y1:bar_y2] = bar_height
            # Place goal just after the hurdle
            goals[i] = [bar_x2 + m_to_idx(0.20), center_y]
            # Allow quadruped some space to pass before next obstacle
            post_bar = gap_base_idx
            cur_x = bar_x2 + post_bar

        # Place stairs
        else:
            # Decide step parameters
            n_steps = np.random.randint(stair_steps_min, stair_steps_max + 1)
            step_depth = m_to_idx(stair_depth_per_step)
            step_rise = stair_rise_per_step
            stair_width = m_to_idx(1.2)
            stair_x1 = cur_x
            stair_x2 = stair_x1 + n_steps * step_depth
            stair_y1 = center_y - stair_width // 2
            stair_y2 = center_y + stair_width // 2
            # Build the steps (ascending)
            for j in range(n_steps):
                sx1 = stair_x1 + j * step_depth
                sx2 = sx1 + step_depth
                sh = (j+1) * step_rise
                height_field[sx1:sx2, stair_y1:stair_y2] = sh
            # Place goal at top step
            goals[i] = [stair_x2 - step_depth//2, center_y]
            # Advance to next
            cur_x = stair_x2 + gap_base_idx

    # Place final goals on last flat area (to ensure 8 in total)
    # If not enough obstacles, put extra goals at the end.
    for j in range(n_obstacles, 8):
        # Place equally spaced final goals at the end of the course
        final_x = min(cur_x + m_to_idx(0.4 * (j - n_obstacles + 1)), m_to_idx(length) - 2)
        goals[j] = [final_x, mid_y]

    # Ensure terrain after last obstacle is flat
    height_field[cur_x:, :] = 0.0

    # Clip any out-of-bounds (in rare case)
    height_field = height_field[:m_to_idx(length), :m_to_idx(width)]

    return height_field, goals

# INSERT TERRAIN FUNCTION DEFINITIONS HERE
