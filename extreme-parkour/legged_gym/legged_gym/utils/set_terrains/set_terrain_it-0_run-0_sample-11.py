import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
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