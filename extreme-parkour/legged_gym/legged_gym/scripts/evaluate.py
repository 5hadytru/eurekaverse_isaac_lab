# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os, time
import sys
from tqdm import tqdm
import argparse
import cv2
from pathlib import Path
import numpy as np
import torch
import cv2
from collections import deque
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
import pickle
import copy
from collections.abc import Sequence

from isaaclab.utils.dict import print_dict
from isaaclab.app import AppLauncher
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import task_registry, add_shared_args, MultiCamVideo, get_camera_coords
from legged_gym.utils.helpers import get_checkpoint


parser = argparse.ArgumentParser()
add_shared_args(parser)

parser.add_argument("--video", action="store_true", default=False, help="Record videos of agent")
parser.add_argument("--num_terrain_types", type=int, help="Number of terrain types. Provided by run_eurekaverse.py and should match the number of set_terrain_fns in the generated set_terrain python file")

parser.add_argument("--checkpoint", type=int, default=-1, help="Which model checkpoint to load. If -1, will load the last checkpoint.")
parser.add_argument("--max_steps", type=int, help="Maximum number of evaluation steps")
parser.add_argument("--use_jit", action="store_true", default=False, help="Load jit script when playing")
parser.add_argument("--metric_granularity", type=str, default="all", choices=["type", "level", "cell", "all"])
parser.add_argument("--no_save", action="store_true", default=False, help="Do not save any evaluation results")
parser.add_argument("--plot_cells", action="store_true", default=False, help="Plot evaluation results in new window")

parser.add_argument("--replay_actions", action="store_true", default=False, help="Replay actions stored from deployment")
parser.add_argument("--replay_depth", action="store_true", default=False, help="Replay depth stored from deployment")

AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()

# if not args.headless:
#     env_cfg, _ = task_registry.get_cfgs(name=args.task)
#     # Setting up exactly one env per terrain grid cell
#     if args.terrain_rows is None:
#         print("Setting terrain_rows to 1 as default")
#         args.terrain_rows = 1
#     if args.terrain_cols is None:
#         args.terrain_cols = env_cfg.terrain.num_cols
#     if args.num_envs is None:
#         args.num_envs = args.terrain_rows * args.terrain_cols

args.script = "evaluate"
if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from legged_gym.envs import *
from isaaclab.sensors import CameraCfg, Camera
import isaaclab.sim as sim_utils

def validate_consecutive_tuples(tuples_list):
    for i, item in enumerate(tuples_list):
        if isinstance(item, tuple):
            # Check if tuple elements are consecutive ascending integers
            for j in range(len(item) - 1):
                if item[j+1] != item[j] + 1:
                    raise AssertionError(f"Tuple at index {i} is not consecutive: {item}")
    return True

def get_camera_cfg(col_idx, row_idx, camera_name, env_id:int):
    """
    Add a camera configuration to the environment config for a specific terrain cell.
    
    Args:
        env_cfg: Environment configuration
        col_idx: Column index in the terrain grid
        row_idx: Row index in the terrain grid
        camera_name: Optional name for the camera (defaults to "cam_r{row}_c{col}")
    """
    cam_offset_dict = get_camera_coords(col_idx, row_idx)

    cam_cfg = CameraCfg(
        prim_path=f"/World/envs/env_{env_id}/{camera_name}",
        update_period=0.1,
        height=272,
        width=544,
        data_types=['rgb'],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=20.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0),
            visible=False
        ),
        offset=CameraCfg.OffsetCfg(
            pos=cam_offset_dict['position'],
            rot=cam_offset_dict['rotation'],  # quaternion (w,x,y,z)
            convention="world"
        )
    )

    return cam_cfg


class SingleEnvCamera(Camera):
    """
    Since each of our cameras is only placed in a single env, we have to overwrite Camera's reset() method,
    which will take all the env_ids given to the DirectRLEnv._reset_idx method since the camera will usually be 
    present in each env
    """
    # def __init__(self, cfg: CameraCfg, env_id:int):
    #     super().__init__(cfg)
    #     self.env_id = env_id

    def reset(self, env_ids: Sequence[int] | None = None):
        env_ids = None # this is the key change

        if not self._is_initialized:
            raise RuntimeError(
                "Camera could not be initialized. Please ensure --enable_cameras is used to enable rendering."
            )
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        # note: cannot do smart indexing here since we do a for loop over data.
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # reset the data
        # note: this recomputation is useful if one performs events such as randomizations on the camera poses.
        self._update_poses(env_ids)
        # Reset the frame count
        self._frame[env_ids] = 0

def evaluate(args):
    faulthandler.enable()

    load_dir = Path(LEGGED_GYM_ROOT_DIR) / "logs" / args.proj_name / args.exptid
    if not load_dir.exists():
        print(f"Error: {load_dir} does not exist!")
        exit()

    try:
        env_cfg, train_cfg = task_registry.get_saved_cfgs(load_dir=load_dir)
        if env_cfg is None or train_cfg is None:
            print("Warning: failed to load saved config, defaulting to current config")
            env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    except:
        # Backwards compatibility
        env_cfg = task_registry.get_saved_cfgs(load_dir=load_dir)
        _, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # If number of environments is too small, increase it to fill up each grid cell (relevant for distillation)
    # env_cfg.scene.num_envs = max(env_cfg.scene.num_envs, env_cfg.terrain.num_rows * env_cfg.terrain.num_cols)

    # single environment for each grid cell
    if args.num_rows is not None:
        env_cfg.scene.num_envs = args.num_rows * args.num_cols
    else:
        env_cfg.scene.num_envs = env_cfg.terrain.num_rows * env_cfg.terrain.num_cols
    assert args.num_envs is None, "Currently hard-coding num_envs to match number of terrain cells"
    env_cfg.depth.camera_num_envs = env_cfg.scene.num_envs
    
    # Don't resample commands during an episode
    env_cfg.commands.resampling_time = 20
    
    # Disable some domain randomization (used to keep friction on)
    # env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    # Disable the curriculum to assign robots evenly across terrain types and difficulties
    env_cfg.terrain.curriculum = False

    # If showing window, allow user to control command velocity to 0
    # This should not be used for headless evaluation since it will affect the results
    if not args.headless:
        env_cfg.commands.ranges.lin_vel_x[0] = 0
        env_cfg.commands.ranges.lin_vel_y[0] = 0

    # prepare environment
    env, _ = task_registry.make_env(args=args, name=args.task, env_cfg=env_cfg, render_mode="rgb_array" if args.video else None)

    obs = env.get_observations()
    max_episode_length = env.max_episode_length
    device = env.device
    rew_term_keys = env.rew_term_sums.keys()

    if args.video:
        # first, get camera names and their column/row assignments
        assert args.num_terrain_types is not None, "Must provide number of terrain types since cameras are evenly distributed among them"
        assert env_cfg.terrain.num_cols % args.num_terrain_types == 0, f"Current camera setup requires equally represented variations (which won't happen here since there are {args.num_terrain_types} terrain types and {env_cfg.terrain.num_cols} columns)"
        print("[INFO] Recording videos during training.")

        camera_col_idxes = list(range(env_cfg.terrain.num_cols))
        camera_row_idxes = [0, 1, 3, 5, 7]

        validate_consecutive_tuples(camera_col_idxes)
        validate_consecutive_tuples(camera_row_idxes)

        cam_names = []
        idx_to_str = lambda idx: str(idx).replace("(", "").replace(")", "").replace(", ", "_") if isinstance(idx, tuple) else str(idx)
        for col_idx in camera_col_idxes: # variation
            for row_idx in camera_row_idxes:
                cam_name = f"cam_{idx_to_str(row_idx)}r_c{idx_to_str(col_idx)}"
                cam_names.append(cam_name)

        print(f"[INFO] Placing {len(cam_names)} cameras")

        video_out_dir = os.path.join(load_dir, "eval_videos")

        # next, map camera name to the env with the robot acting in it (envs that have terrain_types matching the col_idx for the camera + have difficulty matching row_idx)
        def cam_name_to_matched_rows_and_cols(n:str):
            def get_range(nums):
                if "_" in nums:
                    all_values = tuple(map(int, nums.split("_")))
                    return (all_values[0], all_values[-1])
                else:
                    return (int(nums), int(nums))

            inclusive_row_range = get_range(n[4:].split("r")[0])
            inclusive_col_range = get_range(n[4:].split("c")[-1])

            return list(range(inclusive_row_range[0], inclusive_row_range[1] + 1)), list(range(inclusive_col_range[0], inclusive_col_range[1] + 1))

        cam_name_to_env_ids = {}
        for cam_name in cam_names:
            row_range, col_range = cam_name_to_matched_rows_and_cols(cam_name)
            print(cam_name, row_range, col_range)
            mask = torch.isin(env.terrain_types.cpu(), torch.tensor(col_range)) & torch.isin(env.terrain_levels.cpu(), torch.tensor(row_range))
            matched_env_id = int(torch.nonzero(mask).squeeze())

            # print(f"Matched env ID {matched_env_id} with rows {row_range} and cols {col_range} to cam {cam_name}")

            # cameras are no longer tiled; should be matching one env per cam
            assert len(row_range) == 1 and len(col_range) == 1, f"Expected exactly one row and one column for camera {cam_name}, but got {len(row_range)} rows and {len(col_range)} columns"
            
            # now get camera cfg and spawn cam into the scene
            cam_cfg = get_camera_cfg(col_range[0], row_range[0], cam_name, matched_env_id)
            env.scene.sensors[cam_name] = SingleEnvCamera(cam_cfg)

            # we are creating a sensor after scene initialization, so we have to replaicate what
            # the InteractiveScene does upon initializing a sensor
            env.scene.sensors[cam_name]._initialize_impl()
            env.scene.sensors[cam_name]._is_initialized = True

        env = MultiCamVideo(env, video_out_dir, cam_names)

    total_steps = args.max_steps if (args.max_steps is not None and args.max_steps > 0) else 10 * int(max_episode_length)

    # Buffers for metric tracking
    rew_sum_per_env = torch.zeros(env_cfg.scene.num_envs, dtype=torch.float, device=device)
    rew_terms_sum_per_env = {term: torch.zeros(env_cfg.scene.num_envs, dtype=torch.float, device=device) for term in rew_term_keys}
    len_sum_per_env = torch.zeros(env_cfg.scene.num_envs, dtype=torch.float, device=device)
    goals_sum_per_env = torch.zeros(env_cfg.scene.num_envs, dtype=torch.float, device=device)
    sum_counter_per_env = torch.zeros(env_cfg.scene.num_envs, dtype=torch.float, device=device)
    edge_violation_sum_per_env = torch.zeros(env_cfg.scene.num_envs, dtype=torch.float, device=device)

    # cur_rew_sum = torch.zeros(env_cfg.scene.num_envs, dtype=torch.float, device=device)
    cur_episode_length = torch.zeros(env_cfg.scene.num_envs, dtype=torch.float, device=device)
    cur_time_from_start = torch.zeros(env_cfg.scene.num_envs, dtype=torch.float, device=device)

    # Set up loading config
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.exptid
    train_cfg.runner.checkpoint = args.checkpoint

    if args.use_jit:
        policy = torch.jit.load(load_dir / "traced" / "policy_latest.jit").to(device)
        depth_encoder = torch.jit.load(load_dir / "traced" / "depth_latest.jit").to(device)
        # parkour_actor = ParkourActor(device="cuda")
        # parkour_actor.load(load_dir)
        checkpoint = "jit"
    else:
        ppo_runner, train_cfg, _, loaded_dir, checkpoint = task_registry.make_alg_runner(env=env, args=args, name="go1", train_cfg=train_cfg, log_root=load_dir)
        assert load_dir == loaded_dir, f"Config loading directory {load_dir} is different from the runner loading directory {loaded_dir}!"
        if env_cfg.depth.use_camera:
            raise NotImplementedError("Depth actor not ported to Lab")
            policy = ppo_runner.get_depth_actor_inference_policy(device=device)
            if env_cfg.depth.use_camera:
                depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=device)
        else:
            policy = ppo_runner.get_inference_policy(device=device)
    checkpoint_name = checkpoint.replace(".pt", "").replace("_", "-")

    actions = torch.zeros(env_cfg.scene.num_envs, 12, device=device, requires_grad=False)
    if env_cfg.depth.use_camera:
        infos = {
            "depth": env.depth_buffer.clone().cuda()[:, -1]
        }
        depth_latent = None
    
    obs_replay = []
    depth_replay = []
    action_replay = []
    depth_latent_replay = []
    print(f"Running for {total_steps} steps")

    if args.replay_actions:
        saved_actions = np.load(f"{load_dir}/deployed_actions.npy")
        saved_actions = np.tile(saved_actions, (args.num_envs, 1, 1))
        saved_actions = torch.from_numpy(saved_actions).transpose(0, 1)
    
    if args.replay_depth:
        saved_depth = np.load(f"{load_dir}/deployed_depth.npy")
        saved_depth = np.tile(saved_depth, (args.num_envs, 1, 1, 1))
        saved_depth = torch.from_numpy(saved_depth).transpose(0, 1)
    
    for t in tqdm(range(total_steps)):
        if args.replay_actions:
            actions = saved_actions[t % len(saved_actions)]

        elif args.use_jit:
            # Set scandots to 0, should be estimated by Depth Encoder
            lo = env_cfg.env.n_proprio
            hi = lo + env_cfg.env.n_scan
            obs[:, lo:hi] = 0

            # Set privileged explicit to 0, should be estimated by Estimator
            lo = env_cfg.env.n_proprio + env_cfg.env.n_scan
            hi = lo + env_cfg.env.n_priv
            obs[:, lo:hi] = 0

            # Set privileged latents to 0, should be estimated by Actor's history encoder
            lo = env_cfg.env.n_proprio + env_cfg.env.n_scan + env_cfg.env.n_priv
            hi = lo + env_cfg.env.n_priv_latent
            obs[:, lo:hi] = 0

            assert env_cfg.depth.use_camera, "JIT policy is the deployment policy that uses the depth sensor"
            if infos["depth"] is not None:
                depth_replay.append(copy.deepcopy(infos["depth"][0]))
                with torch.no_grad():
                    obs_proprio = obs[:, :env_cfg.env.n_proprio].clone()
                    obs_proprio[5:7] = 0
                    depth_encoder_output = depth_encoder(infos["depth"], obs_proprio)
                    depth_latent_replay.append(copy.deepcopy(depth_encoder_output[0]))
                    if train_cfg.depth_encoder.train_direction_distillation:
                        yaw = depth_encoder_output[:, -2:]
                        depth_latent = depth_encoder_output[:, :-2]
                        if env_cfg.depth.use_direction_distillation:
                            obs[:, 5:7] = 1.5 * yaw
                    else:
                        depth_latent = depth_encoder_output

            with torch.no_grad():
                actions = policy(obs, depth_latent)

            # Save for replay
            obs_replay.append(obs[0:1])
            action_replay.append(actions[0:1])
        else:
            if env_cfg.depth.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, :env_cfg.env.n_proprio].clone()
                    obs_student[:, 5:7] = 0
                    with torch.no_grad():
                        depth_encoder_output = depth_encoder(infos["depth"], obs_student)
                    # depth_latent = depth_latent_and_yaw[:, :-2]
                    # if env_cfg.depth.use_direction_distillation:
                    #     yaw = depth_latent_and_yaw[:, -2:]
                    #     obs[:, 5:7] = 1.5 * yaw

                    if train_cfg.depth_encoder.train_direction_distillation:
                        yaw = depth_encoder_output[:, -2:]
                        depth_latent = depth_encoder_output[:, :-2]
                        if env_cfg.depth.use_direction_distillation:
                            obs[:, 5:7] = 1.5 * yaw
                    else:
                        depth_latent = depth_encoder_output
            else:
                depth_latent = None
            
            if hasattr(ppo_runner.alg, "depth_actor"):
                with torch.no_grad():
                    actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            else:
                actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            
        all_obs, rewards, reset_term, reset_time_out, infos = env.step(actions.detach())
        dones = reset_term | reset_time_out
        obs, _ = all_obs

        if args.replay_depth and t % env_cfg.depth.update_interval == 0:
            infos["depth"] = saved_depth[(t // env_cfg.depth.update_interval) % len(saved_depth)].to(device)

        # Log stuff
        # cur_rew_sum += rews
        cur_rew_sums = infos["rew_sums"]
        cur_reward_term_sums = infos["rew_term_sums"]
        cur_goal_idx = infos["cur_goal_idx"]
        feet_at_edge = env.feet_at_edge.clone().float() if not args.video else env.env.feet_at_edge.clone().float()
        cur_episode_length += 1
        cur_time_from_start += 1

        new_ids = (dones > 0).nonzero(as_tuple=False)[:, 0]
        killed_ids = ((dones > 0) & (~infos["time_outs"])).nonzero(as_tuple=False)[:, 0]

        rew_sum_per_env[new_ids] += cur_rew_sums[new_ids]
        for term in rew_terms_sum_per_env.keys():
            rew_terms_sum_per_env[term][new_ids] += cur_reward_term_sums[term][new_ids]
        len_sum_per_env[new_ids] += cur_episode_length[new_ids]
        goals_sum_per_env[new_ids] += cur_goal_idx[new_ids]
        sum_counter_per_env[new_ids] += 1
        edge_violation_sum_per_env[:] += feet_at_edge.sum(dim=1)

        # cur_rew_sum[new_ids] = 0
        cur_episode_length[new_ids] = 0
        cur_time_from_start[killed_ids] = 0

    if args.use_jit and not args.no_save:
        np.save(f'{load_dir}/action_replay.npy', torch.stack(action_replay).cpu().numpy())
        np.save(f'{load_dir}/obs_replay.npy', torch.stack(obs_replay).cpu().numpy())
        np.save(f'{load_dir}/depth_replay.npy', torch.stack(depth_replay).cpu().numpy())
        np.save(f'{load_dir}/depth_latent_replay.npy', torch.stack(depth_latent_replay).cpu().numpy())
    
    rew_sum_per_env = rew_sum_per_env.cpu()
    rew_terms_sum_per_env = {term: rew_terms_sum_per_env[term].cpu() for term in rew_terms_sum_per_env.keys()}
    len_sum_per_env = len_sum_per_env.cpu()
    goals_sum_per_env = goals_sum_per_env.cpu()
    sum_counter_per_env = sum_counter_per_env.cpu()
    edge_violation_sum_per_env = edge_violation_sum_per_env.cpu()

    # since curriculum=False, terrain_levels (in the LeggedRobot env object) is a (num_envs,) tensor containing
    # a sequence of ranges from 0 to num_rows (number of difficulties; 10 in our case)
    # env_class maps envs to their idxes in the set_terrain list of set_terrain_fns (env's variations)
    # so terrain_cells becomes a set of len num_envs containing tensors like (variation, difficulty)

    # in the case of eurekaverse, the range of variations (set_idx; contained in env_class) will be determined by 
    # config.yaml (it will be range(cfg.num_terrain_types))
    # there will be variations repeated if LeggedRobotCfg.terrain.num_cols > cfg.num_terrain_types
    # the range of difficulties will be range(LeggedRobotCfg.terrain.num_rows)
    env_class = env.env_class if not args.video else env.env.env_class
    terrain_levels = env.terrain_levels if not args.video else env.env.terrain_levels
    terrain_cells = set(zip(env_class.cpu().numpy().tolist(), terrain_levels.cpu().numpy().tolist()))
    mean_rew_per_cell_buffer, mean_rew_terms_per_cell_buffer, mean_len_per_cell_buffer, mean_goals_per_cell_buffer, mean_edge_violation_per_cell_buffer = {}, {}, {}, {}, {}
    mean_rew_terms_per_cell_buffer = {term: {} for term in rew_terms_sum_per_env.keys()}
    sum_counter_per_env[sum_counter_per_env == 0] = 1  # Avoid division by zero
    for cell in terrain_cells:
        terrain_type, terrain_level = cell
        ids = (env_class == terrain_type) & (terrain_levels == terrain_level)
        ids = ids.cpu()
        mean_rew_per_cell_buffer[cell] = torch.sum(rew_sum_per_env[ids]) / torch.sum(sum_counter_per_env[ids])
        for term in rew_terms_sum_per_env.keys():
            mean_rew_terms_per_cell_buffer[term][cell] = torch.sum(rew_terms_sum_per_env[term][ids]) / torch.sum(sum_counter_per_env[ids])
        mean_len_per_cell_buffer[cell] = torch.sum(len_sum_per_env[ids]) / torch.sum(sum_counter_per_env[ids])
        mean_goals_per_cell_buffer[cell] = torch.sum(goals_sum_per_env[ids]) / torch.sum(sum_counter_per_env[ids])
        mean_edge_violation_per_cell_buffer[cell] = torch.sum(edge_violation_sum_per_env[ids]) / torch.sum(sum_counter_per_env[ids])
    
    if not args.no_save:
        pickle_filename = f"{load_dir}/evaluation-{env_cfg.terrain.type}_{checkpoint_name}.pkl"
        if os.path.exists(pickle_filename):
            os.rename(pickle_filename, pickle_filename + ".old")
        with open(pickle_filename, "wb") as f:
            pickle.dump({
                "mean_rew_per_cell_buffer": mean_rew_per_cell_buffer,
                "mean_rew_terms_per_cell_buffer": mean_rew_terms_per_cell_buffer,
                "mean_len_per_cell_buffer": mean_len_per_cell_buffer,
                "mean_goals_per_cell_buffer": mean_goals_per_cell_buffer,
                "mean_edge_violation_per_cell_buffer": mean_edge_violation_per_cell_buffer
            }, f)

    def aggregate_cells(means_per_cell, granularity):
        if granularity == "cell":
            return means_per_cell
        elif granularity == "type":
            means_per_type = {}
            for terrain_type in set([cell[0] for cell in means_per_cell.keys()]):
                cells_in_type = [means_per_cell[cell] for cell in means_per_cell.keys() if cell[0] == terrain_type]
                means_per_type[terrain_type] = np.mean(cells_in_type)
            return means_per_type
        elif granularity == "level":
            means_per_level = {}
            for terrain_level in set([cell[1] for cell in means_per_cell.keys()]):
                cells_in_level = [means_per_cell[cell] for cell in means_per_cell.keys() if cell[1] == terrain_level]
                means_per_level[terrain_level] = np.mean(cells_in_level)
            return means_per_level
        elif granularity == "overall":
            means_all = np.mean(list(means_per_cell.values()))
            return means_all
        else:
            raise ValueError(f"Invalid granularity {granularity}")
    
    results_str = ""
    results_str += "STATISTICS SUMMARY\n"
    rew_mean = aggregate_cells(mean_rew_per_cell_buffer, granularity="overall")
    rew_terms_mean = {term: aggregate_cells(mean_rew_terms_per_cell_buffer[term], granularity="overall") for term in mean_rew_terms_per_cell_buffer.keys()}
    len_mean = aggregate_cells(mean_len_per_cell_buffer, granularity="overall")
    goals_mean = aggregate_cells(mean_goals_per_cell_buffer, granularity="overall")
    edge_violation_mean = aggregate_cells(mean_edge_violation_per_cell_buffer, granularity="overall")
    results_str += f"Reward: {rew_mean:.2f}\n"
    for term in rew_terms_mean.keys():
        results_str += f"Reward term {term}: {rew_terms_mean[term]:.2f}\n"
    results_str += f"Episode length: {len_mean:.2f}\n"
    results_str += f"Number of goals reached: {goals_mean:.2f}\n"
    results_str += f"Edge violation: {edge_violation_mean:.2f}\n"
    results_str += "\n"
    
    granularities = [args.metric_granularity] if args.metric_granularity != "all" else ["cell", "level", "type"]
    for granularity in granularities:
        # Compute mean statistics, weighing over each terrain type and difficulty equally
        # We do this to avoid biasing the results towards harder terrains that cause more resets, which would put more entries in the buffer
        rew_mean_per = aggregate_cells(mean_rew_per_cell_buffer, granularity)
        rew_terms_mean_per = {term: aggregate_cells(mean_rew_terms_per_cell_buffer[term], granularity) for term in mean_rew_terms_per_cell_buffer.keys()}
        len_mean_per = aggregate_cells(mean_len_per_cell_buffer, granularity)
        goals_mean_per = aggregate_cells(mean_goals_per_cell_buffer, granularity)
        edge_violation_mean_per = aggregate_cells(mean_edge_violation_per_cell_buffer, granularity)
        assert rew_mean_per.keys() == len_mean_per.keys() == goals_mean_per.keys() == edge_violation_mean_per.keys(), "Mismatch in keys for statistics"


        granularity_results_str = ""
        for i in sorted(rew_mean_per.keys()):
            if granularity == "cell":
                terrain_type, terrain_level = i
                granularity_results_str += f"STATISTICS FOR TERRAIN TYPE {terrain_type:02}, LEVEL {terrain_level:02}\n"
            elif granularity == "type":
                granularity_results_str += f"STATISTICS FOR TERRAIN TYPE {i:02}\n"
            elif granularity == "level":
                granularity_results_str += f"STATISTICS FOR TERRAIN LEVEL {i:02}\n"
            granularity_results_str += f"Reward: {rew_mean_per[i]:.2f}\n"
            for term in rew_terms_mean_per.keys():
                granularity_results_str += f"Reward term {term}: {rew_terms_mean_per[term][i]:.2f}\n"
            granularity_results_str += f"Episode length: {len_mean_per[i]:.2f}\n"
            granularity_results_str += f"Number of goals reached: {goals_mean_per[i]:.2f}\n"
            granularity_results_str += f"Edge violation: {edge_violation_mean_per[i]:.2f}\n"
            granularity_results_str += "\n"

        # Print and save results
        if args.metric_granularity != "all":
            print(results_str + granularity_results_str)
        if not args.no_save:
            filepath = os.path.join(load_dir, f"evaluation-{env_cfg.terrain.type}_per-{granularity}_{checkpoint_name}.txt")
            if os.path.exists(filepath):
                os.rename(filepath, filepath + ".old")
            with open(filepath, "w", encoding='utf-8') as f:
                f.write(results_str + granularity_results_str)
    
    if "cell" in granularities:
        goals_mean_per = aggregate_cells(mean_goals_per_cell_buffer, "cell")

        num_terrains = torch.unique(env_class).numel()
        num_levels = torch.unique(terrain_levels).numel()
        per_row = min(5, num_terrains)
        fig, axs = plt.subplots(num_terrains // per_row, per_row, figsize=(24, 8))
        keys = sorted(list(goals_mean_per.keys()))
        for i in range(num_terrains):
            ax = axs[i // per_row, i % per_row] if num_terrains > per_row else (axs[i] if num_terrains > 1 else axs)
            means = [goals_mean_per[keys[i * num_levels + x]] for x in range(num_levels)]
            ax.plot(range(num_levels), means)
            ax.axhline(y=1, color='r', linestyle='--')
            ax.axhline(y=8, color='r', linestyle='--')
            ax.set_xlim(0, num_levels-1)
            ax.set_ylim(0, 8.5)
            ax.set_title(i)
        plt.tight_layout()

        if not args.no_save:
            save_filename = f"{load_dir}/evaluation-{env_cfg.terrain.type}_{checkpoint_name}.png"
            if os.path.exists(save_filename):
                os.rename(save_filename, save_filename + ".old")
            plt.savefig(save_filename)
        if args.plot_cells:
            plt.show()

    env.close()
    try:
        env.env.close()
    except:
        pass

if __name__ == '__main__':
    evaluate(args)
    simulation_app.close()
