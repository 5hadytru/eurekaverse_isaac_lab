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

import os
import copy
import torch
import numpy as np
import random
import argparse

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_checkpoint(load_dir, checkpoint=-1, model_name_include="model"):
    if checkpoint == -1:
        # Get last checkpoint
        models = [file for file in os.listdir(load_dir) if model_name_include in file and ".pt" in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = f"model_{checkpoint}.pt"
    return model


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        if args.device is not None:
            env_cfg.sim.device = args.device
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed
        if args.terrain_type is not None:
            env_cfg.terrain.type = args.terrain_type
        if args.check_terrain_feasibility:
            env_cfg.terrain.check_feasibility = True
        
        if hasattr(args, "video"):
            env_cfg.video = args.video
        else:
            env_cfg.video = False

        if args.terrain_type == "simple": # flat
            env_cfg.commands.ranges.lin_vel_x = [0.0, 2.0]
        else: # parkour
            env_cfg.commands.ranges.lin_vel_x = [0.3, 1.2]

        # terrain-related args
        if args.terrain_length is not None:
            env_cfg.terrain.terrain_length = args.terrain_length
        if args.terrain_width is not None:
            env_cfg.terrain.terrain_width = args.terrain_width
        if args.vertical_scale is not None:
            env_cfg.terrain.vertical_scale = args.vertical_scale
        if args.horizontal_scale is not None:
            env_cfg.terrain.horizontal_scale = args.horizontal_scale
        if args.num_cols is not None:
            env_cfg.terrain.num_cols = args.num_cols
        if args.num_rows is not None:
            env_cfg.terrain.num_rows = args.num_rows
        if args.num_goals is not None:
            env_cfg.terrain.num_goals = args.num_goals

    if cfg_train is not None and args.script == "train":
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
            # If we're resuming, we set the privileged regression loss coefficient to max value immediately
            cfg_train.algorithm.priv_reg_coef_schedual = cfg_train.algorithm.priv_reg_coef_schedual_resume
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def add_sim_args(parser):
    # Simulation setup
    parser.add_argument("--proj_name", type=str, default="parkour", help="Main project name, used for logging and saving")
    parser.add_argument("--exptid", type=str, help="Experiment name, used for logging and saving")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Weights and Biases tracking
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use wandb for logging")
    parser.add_argument("--wandb_id", type=str, help="Wandb run id, used to continue another run")
    parser.add_argument("--wandb_group", type=str, help="Wandb group name, used to group runs")

def add_agent_args(parser):
    parser.add_argument("--task", type=str, default="go1", help="Which robot to use")
    parser.add_argument("--num_envs", type=int, help="Number of environments to create")
    parser.add_argument("--use_camera", action="store_true", default=False, help="Render camera for distillation")

def add_terrain_args(parser):
    parser.add_argument("--terrain_type", type=str, default="default", help="Which set_terrain() function file to use")
    parser.add_argument("--terrain_length", type=int, help="")
    parser.add_argument("--terrain_width", type=int, help="")
    parser.add_argument("--vertical_scale", type=float, help="")
    parser.add_argument("--horizontal_scale", type=float, help="")
    parser.add_argument("--num_cols", type=int, help="")
    parser.add_argument("--num_rows", type=int, help="")
    parser.add_argument("--num_goals", type=int, help="")
    parser.add_argument("--check_terrain_feasibility", action="store_true", default=False, help="Check terrain feasibility with simple heuristics")

def add_shared_args(parser):
    add_sim_args(parser)
    add_agent_args(parser)
    add_terrain_args(parser)

def export_policy_as_jit(actor_critic, path, name):
    if hasattr(actor_critic, 'memory_a'):
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, name+".pt")
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)