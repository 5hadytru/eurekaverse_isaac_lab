import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
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