import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Slalom-style course using a series of alternating wide barriers for lateral maneuvering skill."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        if isinstance(m, (list, tuple)):
            return [round(i / field_resolution) for i in m]
        return np.round(m / field_resolution).astype(np.int16)

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- COURSE PLAN ---
    # The course consists of 6 alternating wide barriers ("slalom poles"), spaced along x.
    # Barriers reach from one side of the course to the other except for a wide gap (the 'gate') on alternating L/R sides.
    # The quadruped must alternate its y-position while advancing, requiring agile lateral movement.
    # Barrier height increases with difficulty.
    # Finishes with a straight sprint to the final goal.

    # Obstacle and gap sizing
    barrier_width = 0.2 + 0.2 * difficulty      # meters, thickness of each slalom barrier
    barrier_height = 0.1 + 0.25 * difficulty    # meters, climbs at low difficulty, jumps/bar jumps at high
    gate_width = 1.2 - 0.5 * difficulty         # meters, narrows with difficulty

    # Ensure gaps are at least quadruped body width even at max difficulty
    gate_width = max(gate_width, 0.4)
    # Spacing between barriers
    n_barriers = 6
    barrier_spacing = (length - 3) / (n_barriers + 1)  # leave space for spawn (2m) and finish (1m)

    mid_y = m_to_idx(width / 2)
    spawn_x = m_to_idx(1.0)
    spawn_area = m_to_idx(2.0)

    # First goal is just ahead of spawn, centered
    goals[0] = [spawn_x + m_to_idx(0.4), mid_y]

    # Set spawn area to flat
    height_field[:spawn_area, :] = 0

    # Place alternating barriers and path goals
    for i in range(n_barriers):
        x0_m = 2.0 + (i + 1) * barrier_spacing
        x0 = m_to_idx(x0_m)
        barrier_thick = m_to_idx(barrier_width)
        gate = m_to_idx(gate_width)
        y_total = m_to_idx(width)
        side = i % 2  # 0: left gate, 1: right gate

        # Compute where the gap (gate) is
        if side == 0:  # gap on left
            gate_start = 0
            gate_end = gate
        else:  # gap on right
            gate_start = y_total - gate
            gate_end = y_total

        # Fill most of the barrier except for the gate
        if side == 0:
            # barrier occupies [gate_end, end]
            height_field[x0:x0 + barrier_thick, gate_end:] = barrier_height
        else:
            # barrier occupies [0, gate_start]
            height_field[x0:x0 + barrier_thick, :gate_start] = barrier_height

        # Place goal just past the barrier, at the center of the gate
        x_goal = x0 + barrier_thick + m_to_idx(0.25)  # just after barrier
        if side == 0:
            y_goal = (gate_start + gate_end) // 2
        else:
            y_goal = (gate_start + gate_end) // 2
        goals[i + 1] = [x_goal, y_goal]

    # Final stretch: lets the robot run straight to a final goal
    finish_x = m_to_idx(length - 1.0)
    goals[7] = [finish_x, mid_y]

    # Ensure the finish is flat
    height_field[finish_x:, :] = 0

    return height_field, goals