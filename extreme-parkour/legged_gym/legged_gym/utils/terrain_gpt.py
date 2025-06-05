import os
import numpy as np
from numpy.random import choice
from scipy import interpolate
from math import sqrt
import random
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, CustomTerrainCfg
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation
import inspect
import uuid
import importlib.util
from legged_gym.utils.helpers import set_seed
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf
from datetime import datetime

from isaaclab.terrains.utils import create_prim_from_mesh
import trimesh
import torch

# This is the standard set_terrain
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.terrain_utils import fix_terrain, calc_direct_path_heights
from legged_gym.utils.set_terrain import set_terrain as set_terrain
from legged_gym.utils.set_terrain_demo import set_terrain as set_terrain_demo
from legged_gym.utils.set_terrain_benchmark import set_terrain as set_terrain_benchmark
from legged_gym.utils.set_terrain_original import set_terrain as set_terrain_original
from legged_gym.utils.set_terrain_original_distill import set_terrain as set_terrain_original_distill
from legged_gym.utils.set_terrain_presets import set_terrain as set_terrain_presets
from legged_gym.utils.set_terrain_simple import set_terrain as set_terrain_simple
from legged_gym.utils.set_terrain_platforms import set_terrain as set_terrain_platforms
from legged_gym.utils.set_terrain_random import set_terrain as set_terrain_random
from legged_gym.utils.set_terrain_real import set_terrain as set_terrain_real

# Override default set_terrain.py with a custom path
set_terrain_override = None
# set_terrain_override = "/home/exx/Projects/eurekaverse/eurekaverse/outputs/eurekaverse/2024-05-26_22-32-38/terrain_iter-4_run-3.py"

class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)

def load_terrain_function_from_file(filepath):
    spec = importlib.util.spec_from_file_location("module_name", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = module.set_terrain
    return function

def run_ambiguous_set_terrain(set_terrain_fn, terrain, variation, difficulty):
    signature = inspect.signature(set_terrain_fn)
    args = [p.name for p in signature.parameters.values()]
    if set(args) == set(["terrain", "variation", "difficulty"]):
        set_idx = set_terrain_fn(terrain, variation, difficulty)
    elif set(args) == set(["terrain", "difficulty"]):
        set_idx = set_terrain_fn(terrain, difficulty)
    elif set(args) == set(["length", "width", "field_resolution", "difficulty"]):
        height_field, goals = set_terrain_fn(terrain.width * terrain.horizontal_scale, terrain.length * terrain.horizontal_scale, terrain.horizontal_scale, difficulty)
        terrain.height_field_raw = (height_field / terrain.vertical_scale).astype(np.int16)
        terrain.goals = goals
        set_idx = None
    else:
        raise ValueError(f"set_terrain function signature not recognized: {args}")
    return set_idx

class TrimeshTerrainImporter:
    def __init__(
        self, 
        vertices:np.ndarray,
        triangles:np.ndarray,
        translation,
        initial_env_origins:torch.Tensor, 
        physics_material_cfg, 
        visual_material_cfg,
        device
    ):
        self.vertices = vertices
        self.triangles = triangles
        self.translation = translation
        self.env_origins = initial_env_origins.clone()
        self.visual_material = visual_material_cfg
        self.physics_material = physics_material_cfg
        self.device = device

        self.prim_path = "/World/Terrain"

        self.import_mesh()

    def import_mesh(self):
        """
        Create a trimesh.Trimesh object -> store as USD prim
        """
        mesh = trimesh.Trimesh(self.vertices, self.triangles)
        create_prim_from_mesh(
            self.prim_path, mesh, visual_material=self.visual_material, physics_material=self.physics_material, translation=self.translation
        )

class Terrain:
    def __init__(self, cfg: CustomTerrainCfg, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        # cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        # self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.int64)
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
        # self.num_goals = cfg.num_goals

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        if set_terrain_override is not None and self.cfg.type == "default":
            print(f"Warning: Using set_terrain override, getting terrain from {set_terrain_override}")
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / (self.cfg.num_rows-1) if self.cfg.num_rows > 1 else 0.5
                variation = j / self.cfg.num_cols
                terrain = self.make_terrain(variation, difficulty)
                
                # Pad borders
                pad_width = int(0.1 // terrain.horizontal_scale)
                pad_height = int(0.5 // terrain.vertical_scale)
                terrain.height_field_raw[:, :pad_width] = pad_height
                terrain.height_field_raw[:, -pad_width:] = pad_height
                terrain.height_field_raw[:pad_width, :] = pad_height
                terrain.height_field_raw[-pad_width:, :] = pad_height

                self.add_terrain_to_map(terrain, i, j)

        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            print("Converting heightmap to trimesh...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                                self.cfg.horizontal_scale,
                                                                                                self.cfg.vertical_scale,
                                                                                                self.cfg.slope_treshold)
                half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
                structure = np.ones((half_edge_width*2+1, 1))
                self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)
                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(target_count = int(0.05*self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert cfg.hf2mesh_method == "fast", "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, max_error=cfg.max_error)
            
            print(f"Created vertices {self.vertices.shape}")
            print(f"Created triangles {self.triangles.shape}")

            self.vertices, self.triangles = self.vertices.tolist(), self.triangles.tolist()

    def make_terrain(self, variation, difficulty):
        # Make terrain generation deterministic
        # NOTE: The seed will be reset back to env_cfg.seed after the environment is created, inside TaskRegistry.make_env()
        set_seed(int(variation * 1e3 + difficulty * 1e6))
        # NOTE: Width and length are swapped in the terrain_utils.SubTerrain, careful!
        terrain = SubTerrain(
            "terrain",
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale
        )
        terrain.goals = np.zeros((self.cfg.num_goals, 2))

        fix_desc = ""
        if self.cfg.type == "default":
            if set_terrain_override is not None:
                set_terrain_fn = load_terrain_function_from_file(set_terrain_override)
            else:
                set_terrain_fn = set_terrain
            set_idx = run_ambiguous_set_terrain(set_terrain_fn, terrain, variation, difficulty)
            fix_desc = fix_terrain(terrain)
            if self.cfg.check_feasibility:
                check_terrain_feasibility(terrain, allow_flat_terrain=(difficulty == 0))
        elif self.cfg.type == "demo":
            set_idx = set_terrain_demo(terrain, variation, difficulty)
        elif self.cfg.type == "benchmark":
            set_idx = set_terrain_benchmark(terrain, variation, difficulty)
        elif self.cfg.type == "original":
            set_idx = set_terrain_original(terrain, variation, difficulty)
        elif self.cfg.type == "original_distill":
            set_idx = set_terrain_original_distill(terrain, variation, difficulty)
        elif self.cfg.type == "presets":
            set_idx = set_terrain_presets(terrain, variation, difficulty)
        elif self.cfg.type == "simple":
            set_idx = set_terrain_simple(terrain, variation, difficulty)
        elif self.cfg.type == "platforms":
            set_idx = set_terrain_platforms(terrain, variation, difficulty)
        elif self.cfg.type == "random":
            set_idx = set_terrain_random(terrain, variation, difficulty)
        elif self.cfg.type == "real":
            set_idx = set_terrain_real(terrain, variation, difficulty)
        else:
            if self.cfg.type == "test":
                filepath = f"{LEGGED_GYM_ROOT_DIR}/legged_gym/utils/set_terrain_test.py"
            else:
                filepath = f"{LEGGED_GYM_ROOT_DIR}/legged_gym/utils/set_terrains/set_terrain_{self.cfg.type}.py"
            if not os.path.exists(filepath):
                raise ValueError(f"Terrain type {self.cfg.type} not recognized!")
            set_terrain_fn = load_terrain_function_from_file(filepath)
            set_idx = run_ambiguous_set_terrain(set_terrain_fn, terrain, variation, difficulty)
            fix_desc = fix_terrain(terrain)
            if self.cfg.check_feasibility:
                check_terrain_feasibility(terrain, allow_flat_terrain=(difficulty == 0))
        terrain.idx = set_idx if set_idx is not None else 0
        if fix_desc != "":
            print(f"Automatically fixed terrain {terrain.idx}: {fix_desc}")

        # Add roughness to terrain
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * 0.5 + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain = random_uniform_terrain(terrain, min_height=-height, max_height=height, step=self.cfg.vertical_scale, downsampled_scale=self.cfg.downsampled_scale)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((1.0 - 0.5) / terrain.horizontal_scale) # within 1 meter square range
        x2 = int((1.0 + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2 - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2 + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = terrain.idx
        self.goals[i, j, :, :2] = terrain.goals + [i * self.env_length, j * self.env_width]

def check_terrain_feasibility(terrain, allow_flat_terrain=False):
    max_terrain_height = np.max(terrain.height_field_raw)
    assert max_terrain_height <= round(3 / terrain.vertical_scale), f'Generated terrain with maximum height {max_terrain_height} exceeds height bound!'
    start_location = np.array([2, (terrain.length / 2 * terrain.horizontal_scale)])
    goals = np.concatenate([start_location[None, :], terrain.goals], axis=0) / terrain.horizontal_scale
    _, heights = calc_direct_path_heights(terrain.height_field_raw, goals, skip_size=round(1 / terrain.horizontal_scale))
    heights = [i for sublist in heights for i in sublist]
    diff_along_path = np.max(np.abs(np.diff(heights)))
    assert diff_along_path <= round(0.8 / terrain.vertical_scale), f'Generated terrain has maximum height difference of {diff_along_path} along direct path, not feasible!'
    if not allow_flat_terrain:
        assert diff_along_path > 0, 'Generated terrain has no height difference along direct path, no challenge!'


def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    # Modified from isaacgym.terrain_utils.convert_heightfield_to_trimesh to also return x_edge_mask

    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0

def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None,):
    """
    Generate a uniform noise terrain

    Parameters
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    # switch parameters to discrete units
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(heights_range, (int(terrain.width * terrain.horizontal_scale / downsampled_scale), int(
        terrain.length * terrain.horizontal_scale / downsampled_scale)))

    x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

    f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

    x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    terrain.height_field_raw += z_upsampled.astype(np.int16)
    return terrain

