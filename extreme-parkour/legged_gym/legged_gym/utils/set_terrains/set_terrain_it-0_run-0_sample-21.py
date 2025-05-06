import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Slalom course: Tall narrow walls form a zig-zag corridor to train precise turning and navigation."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- Parameters ---
    # Walls: vertical obstacles, narrow (0.2-0.4m thick), spanning nearly the full course width, placed at intervals like slalom gates
    # Robot must zig-zag between them, requiring sharp turns and precise foot placement
    # Difficulty scales: wall height, wall thickness, slalom angle (corridor offset), corridor width

    n_walls = 6
    min_wall_spacing = 1.5          # meters between "gates"
    max_wall_spacing = 2.2
    wall_spacing = min_wall_spacing + (max_wall_spacing - min_wall_spacing) * (1 - difficulty)
    wall_thickness = 0.18 + 0.14 * difficulty       # Thicker at higher difficulty
    wall_height = 0.15 + 0.5 * difficulty           # Up to 0.65m, harder at higher diffs
    clear_path_width = 1.2 - 0.5 * difficulty       # Path gets narrower as difficulty increases

    # Spawn and course dimensions
    spawn_x = m_to_idx(1)
    spawn_y = m_to_idx(width/2)
    start_x = m_to_idx(2)
    course_length = m_to_idx(length)
    course_width = m_to_idx(width)
    field_mid_y = course_width // 2

    # Place initial spawn and goal
    height_field[:start_x, :] = 0  # Keep spawn area clear
    goals[0] = [spawn_x, field_mid_y]

    # Set walls at staggered y-offsets to create a zig-zag
    # Walls alternate opening left/right around midline; offset amplitude scales with difficulty
    zigzag_amplitude = (0.7 + 0.6 * difficulty) * (width/2 - clear_path_width/2 - 0.1)
    slalom_angle = np.pi/8 + (np.pi/10 * difficulty)  # range and curvature

    # Compute wall x-positions (from just after spawn)
    wall_xs = [
        start_x + i * m_to_idx(wall_spacing)
        for i in range(n_walls)
    ]

    # Compute wall center y-offsets in a sinusoidal slalom pattern
    wall_ys = [
        int(field_mid_y + np.sin(i * slalom_angle) * m_to_idx(zigzag_amplitude))
        for i in range(n_walls)
    ]

    wall_hw = m_to_idx(wall_thickness/2)
    path_hw = m_to_idx(clear_path_width/2)

    # Place walls and assign goals just in front of each wall (after turning)
    x_prev = spawn_x
    y_prev = field_mid_y

    for i, (wx, wy) in enumerate(zip(wall_xs, wall_ys)):
        # Walls span nearly the full width, except leave corridor clear at the offset
        # Walls are "vertical" in x, blocks full y except within the gap
        wall_y1 = max(0, wy - path_hw - wall_hw)
        wall_y2 = min(course_width, wy + path_hw + wall_hw)
        gap_y1 = wy - path_hw
        gap_y2 = wy + path_hw

        # Fill wall except the corridor gap (gap_y1:gap_y2)
        height_field[wx:wx+m_to_idx(wall_thickness), 0:gap_y1] = wall_height
        height_field[wx:wx+m_to_idx(wall_thickness), gap_y2:course_width] = wall_height

        # The robot should aim just prior to (and centered on) each gap; set as goal
        # Place goal about 0.7m before the center of the gap, so robot is "forced" to zig-zag
        approach_dist = m_to_idx(0.75)
        approach_x = wx - approach_dist if (wx - approach_dist > 0) else max(wx-1, 0)
        goals[i+1] = [approach_x, wy]

        x_prev, y_prev = wx, wy

    # Final straight clear section, final goal at the end center
    height_field[wall_xs[-1] + m_to_idx(wall_thickness):, :] = 0
    goals[7] = [course_length-m_to_idx(1), field_mid_y]

    return height_field, goals