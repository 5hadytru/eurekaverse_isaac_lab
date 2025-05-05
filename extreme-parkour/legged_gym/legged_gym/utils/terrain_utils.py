import os
import numpy as np
from numpy.random import choice
from scipy import interpolate
from math import sqrt
import random
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation
import inspect
import uuid
import importlib.util
from datetime import datetime
import torch

def fix_terrain(terrain):
    """Fix common errors with GPT-generated terrains"""
    # If goals are in units (indices), convert to meters
    # This doesn't count as a fix since we prompt GPT to return goals in units (for simplicity)
    env_length, env_width = terrain.width * terrain.horizontal_scale, terrain.length * terrain.horizontal_scale
    if np.max(terrain.goals[:, 0]) > env_length or np.max(terrain.goals[:, 1]) > env_width:
        terrain.goals = terrain.goals.astype(np.float64) * terrain.horizontal_scale
    
    fix_descs = set()

    min_terrain_height = np.min(terrain.height_field_raw)
    if min_terrain_height < round(-1 / terrain.vertical_scale):
        terrain.height_field_raw[terrain.height_field_raw < -1] = round(-1 / terrain.vertical_scale)
        fix_descs.add(f"min terrain height {min_terrain_height} is below -1")

    # Fix goals that are unset or out of bounds
    def valid_goal(goal):
        return 0 < goal[0] < env_length and 0 < goal[1] < env_width  # We check > 0 since (0, 0) is the default
    num_goals_fixed = 0
    for i in range(1, len(terrain.goals)):
        if not valid_goal(terrain.goals[i]) and valid_goal(terrain.goals[i-1]):
            terrain.goals[i] = terrain.goals[i-1]
            num_goals_fixed += 1
    for i in range(len(terrain.goals) - 2, -1, -1):
        if not valid_goal(terrain.goals[i]) and valid_goal(terrain.goals[i+1]):
            terrain.goals[i] = terrain.goals[i+1]
            num_goals_fixed += 1
    if num_goals_fixed > 0:
        fix_descs.add(f"{num_goals_fixed} goal(s) out of bounds")
    assert num_goals_fixed <= round(len(terrain.goals) / 2), f'Fixed too many goals ({num_goals_fixed})!'
    for i in range(len(terrain.goals)):
        assert valid_goal(terrain.goals[i]), f'Goal {i} at ({terrain.goals[i, 0]}, {terrain.goals[i, 1]}) is invalid!'

    # Move goals away from edge
    clipped_goals_x = np.clip(terrain.goals[:, 0], a_min=0.5, a_max=(env_length - 0.5))
    clipped_goals_y = np.clip(terrain.goals[:, 1], a_min=0.5, a_max=(env_width - 0.5))
    if not np.allclose(clipped_goals_x, terrain.goals[:, 0]) or not np.allclose(clipped_goals_y, terrain.goals[:, 1]):
        fix_descs.add("goals too close to edge")
    terrain.goals[:, 0] = clipped_goals_x
    terrain.goals[:, 1] = clipped_goals_y

    # Check and fix quadruped's spawn location
    if np.max(terrain.height_field_raw[:round(2 / terrain.horizontal_scale), :]) > 0:
        terrain.height_field_raw[:round(2 / terrain.horizontal_scale), :] = 0
        fix_descs.add("spawn area not 0")
    clipped_goals_x = np.clip(terrain.goals[:, 0], a_min=1.5, a_max=None)  # Move goals ahead of spawn
    if not np.allclose(clipped_goals_x, terrain.goals[:, 0]):
        fix_descs.add("goals too close to spawn")
    terrain.goals[:, 0] = clipped_goals_x

    # Check and fix small obstacles that have an extreme aspect ratio
    # This only works for axis-aligned obstacles, but the mistake is rare enough to not warrant a more complex fix
    min_terrain_height = np.min(terrain.height_field_raw)
    valid_ratio_threshold = 2
    min_obstacle_length, min_obstacle_width = 0.6 / terrain.horizontal_scale, 0.4 / terrain.horizontal_scale
    floodfill_dz_threshold = 1 / terrain.vertical_scale
    obstacles = {}
    floodfill = np.zeros_like(terrain.height_field_raw)

    def bfs(x, y, id):
        q = [(x, y)]
        while len(q) > 0:
            x, y = q.pop(0)
            if floodfill[x, y] != 0:
                continue
            floodfill[x, y] = id
            obstacles[id] = [
                (min(obstacles[id][0][0], x), min(obstacles[id][0][1], y)),
                (max(obstacles[id][1][0], x+1), max(obstacles[id][1][1], y+1))
            ]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < terrain.height_field_raw.shape[0] and 0 <= ny < terrain.height_field_raw.shape[1]:
                    if terrain.height_field_raw[nx, ny] != min_terrain_height and floodfill[nx, ny] == 0 and abs(terrain.height_field_raw[nx, ny] - terrain.height_field_raw[x, y]) < floodfill_dz_threshold:
                        q.append((nx, ny))
    obstacle_counter = 0
    for i in range(terrain.height_field_raw.shape[0]):
        for j in range(terrain.height_field_raw.shape[1]):
            if terrain.height_field_raw[i, j] != min_terrain_height and floodfill[i, j] == 0:
                obstacle_counter += 1
                obstacles[obstacle_counter] = [(i, j), (i, j)]
                bfs(i, j, obstacle_counter)
    
    for obstacle in obstacles:
        x1, y1 = obstacles[obstacle][0]
        x2, y2 = obstacles[obstacle][1]
        obstacle_length, obstacle_width = x2 - x1, y2 - y1
        if max(obstacle_length, obstacle_width) / min(obstacle_width, obstacle_length) < valid_ratio_threshold:
            continue
        
        if obstacle_length < min_obstacle_length and obstacle_width < min_obstacle_width:
            # Erase small obstacles
            terrain.height_field_raw[x1:x2, y1:y2] = 0
            fix_descs.add("obstacles length and width too small (erased)")
        if obstacle_length < min_obstacle_length:
            # Extend length on both sides
            extend_length = max(round((min_obstacle_length - obstacle_length) // 2), 1)
            nx1, nx2 = max(0, x1 - extend_length), min(terrain.height_field_raw.shape[0], x2 + extend_length)
            terrain.height_field_raw[nx1:x1, y1:y2] = terrain.height_field_raw[x1, y1:y2][None, :]
            terrain.height_field_raw[x2:nx2, y1:y2] = terrain.height_field_raw[x2-1, y1:y2][None, :]
            fix_descs.add("obstacles length too small")
        if obstacle_width < min_obstacle_width:
            # Extend width on both sides
            extend_width = max(round((min_obstacle_width - obstacle_width) // 2), 1)
            ny1, ny2 = max(0, y1 - extend_width), min(terrain.height_field_raw.shape[1], y2 + extend_width)
            terrain.height_field_raw[x1:x2, ny1:y1] = terrain.height_field_raw[x1:x2, y1][..., None]
            terrain.height_field_raw[x1:x2, y2:ny2] = terrain.height_field_raw[x1:x2, y2-1][..., None]
            fix_descs.add("obstacles width too small")
    
    return ", ".join(fix_descs)


def calc_direct_path_heights(height_field_raw, goals, skip_size):
    """Runs Bresenham's line algorithm to check heights along direct path between goals."""
    # NOTE: goals is in indices, not meters

    all_line_heights = []
    all_skip_line_heights = []
    for i in range(len(goals) - 1):
        (goal_x, goal_y), (next_goal_x, next_goal_y) = goals[i], goals[i + 1]
        goal_x, goal_y, next_goal_x, next_goal_y = round(goal_x), round(goal_y), round(next_goal_x), round(next_goal_y)

        dx, dy = abs(next_goal_x - goal_x), abs(next_goal_y - goal_y)
        sx, sy = 1 if goal_x < next_goal_x else -1, 1 if goal_y < next_goal_y else -1
        err = dx - dy

        # Extract height along the line
        x, y = goal_x, goal_y
        line_heights = [height_field_raw[x, y]]
        while x != next_goal_x or y != next_goal_y:
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            line_heights.append(height_field_raw[x, y])
        all_line_heights.append(line_heights)
        
        # Check max height difference in line_heights
        # We must also account for gap obstacles: a large height difference is
        # allowed if there is a platform with smaller height difference right after
        j = 0
        skip_line_heights = []
        while j < len(line_heights) - 1:
            skip_line_heights.append(line_heights[j])
            k = min(j + skip_size + 1, len(line_heights))
            diff_along_range = line_heights[j+1:k] - line_heights[j]      # Difference between jump destinations (i+1:j) and jump origin (i)
            diff_along_range = np.maximum.accumulate(diff_along_range)    # Every point is at least as high as the points before
                                                                          # This fills up gap unless it starts at i+1, and it also prevents edge cases with walls
            diff_along_range = np.abs(diff_along_range)
            min_diff_idx = np.argmin(diff_along_range)                    # Find optimal jump destination
            j += min_diff_idx + 1                                         # Move to next jump destination
        skip_line_heights.append(line_heights[-1])
        all_skip_line_heights.append(skip_line_heights)

    return all_line_heights, all_skip_line_heights  # First list is for all line heights, second list is with considering skips