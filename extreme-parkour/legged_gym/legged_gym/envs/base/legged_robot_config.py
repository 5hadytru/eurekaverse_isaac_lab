from isaaclab.utils.configclass import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg

@configclass
class LeggedRobotCfg(DirectRLEnvCfg):
    def __post_init__(self):
        """Initialize all member classes recursively"""
        self._init_member_classes(self)
    
    @staticmethod
    def _init_member_classes(obj):
        import inspect
        for key in dir(obj):
            if key.startswith("__"):
                continue
            var = getattr(obj, key)
            if inspect.isclass(var):
                i_var = var()
                setattr(obj, key, i_var)
                LeggedRobotCfg._init_member_classes(i_var)

    sim: SimulationCfg = SimulationCfg(
        device = "cuda" # can be "cpu", "cuda", "cuda:<device_id>"
        dt = 0.005,
        # decimation will be set in the task config
        # up axis will always be Z in isaac sim
        # use_gpu_pipeline is deduced from the device
        gravity=(0.0, 0.0, -9.81),
        up_axis = 1,  # 0 is y, 1 is z
        physx: PhysxCfg = PhysxCfg(
            # num_threads is no longer needed
            solver_type=1,
            # use_gpu is deduced from the device
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            # moved to actor config
            # moved to actor config
            bounce_threshold_velocity=0.5,
            # moved to actor config
            # default_buffer_size_multiplier is no longer needed
            gpu_max_rigid_contact_count=2**24,
            # num_subscenes is no longer needed
            # contact_collection is no longer needed
            contact_offset = 0.01,  # [m]
            rest_offset = 0.0,   # [m]
            max_depenetration_velocity = 1.0,
    )),
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs = 6144,
        env_spacing=3.0
    ),

    # REQUIRED TOP-LEVEL ENV ARGS
    decimation = 1
    episode_length_s = 20 # episode length in seconds
    action_space = 12
    observation_space = n_proprio + n_scan + history_len * n_proprio + n_priv_latent + n_priv
    state_space = 0 # hopefully this is correct

    class env:
        n_scan = 132                             # Scandots
        n_priv = 3 + 3 + 3                       # Privileged (explicit): base linear velocity
        n_priv_latent = 4 + 1 + 12 + 12          # Privileged (latent): mass, friction coefficients, motor strength (Kp and Kd factors, only used for control_type = "P")
        n_proprio = 3 + 2 + 2 + 1 + 24 + 12 + 4  # Proprioceptive: base angular velocity, roll & pitch (imu), delta yaw to curent goal & next goal, velocity command, joint positions & velocities, previous actions, foot contacts
        history_len = 10

        # num_observations = n_proprio + n_scan + history_len * n_proprio + n_priv_latent + n_priv
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        # num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        # episode_length_s = 20 # episode length in seconds
        
        history_encoding = True
        reorder_dofs = True

        include_foot_contacts = True
        
        randomize_start_pos = False
        randomize_start_vel = False
        randomize_start_yaw = False
        rand_yaw_range = 1.2
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = False
        rand_pitch_range = 1.6

        contact_buf_len = 100

        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2

        render_envs = False  # Set up cameras to render each env

    class depth:
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        # Camera setup from Extreme Parkour
        # This is a front-facing camera
        # position = {
        #     "mean": [0.27, 0, 0.03],
        #     "std": [0, 0, 0]
        # }
        # rotation = {  # Note: originally uniformly randomized [-5, 5]
        #     "mean": [0, 0, 0],
        #     "std": [0, 2.5, 0]
        # }

        # Camera setup from Parkour Learning
        # This is a front-facing camera mounted externally, slightly tilted down
        position = {
            "mean": [0.245 + 0.027, 0.0075, 0.072 + 0.02],
            "std": [0.002, 0.002, 0.0002]
        }
        rotation = {
            "mean": [0, 0.52, 0],
            "std": [0, 0.01, 0]
        }

        update_interval = 5        # Update depth every n steps (50Hz for policy dt at 0.02s)
        depth_delay_steps = 1      # Add latency of n steps (1 step is 0.02s)
        depth_buf_len = depth_delay_steps + 1

        # Original resolution for Realsense D435 is 640 x 360, we scale down by factor of 6
        original_resolution = (int(640 / 6), int(360 / 6))
        # We crop out the noisy edges of the image to get the processed resolution
        crop_top, crop_bottom, crop_left, crop_right = 0, 0, 8, 8  # In terms of the scaled-down resolution
        # crop_top, crop_bottom, crop_left, crop_right = 4, 0, 8, 8  # In terms of the scaled-down resolution
        processed_resolution = (original_resolution[0] - crop_left - crop_right, original_resolution[1] - crop_top - crop_bottom)

        horizontal_fov = 87

        near_clip = 0
        far_clip = 2
        bias_noise = 0.0
        granular_noise = 0.02
        blackout_noise = 0.03
        blur_prob = 0.0
        erase_prob = 0.0
        erase_size = [5, 20]

        use_direction_distillation = False

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 1.2
    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        quantize_height = True
        class noise_scales:
            rotation = 0.0
            dof_pos = 0.01
            dof_vel = 0.05
            lin_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02
            height_measurements = 0.02

    class terrain:
        type = "default"  # Which set_terrain() function to use
        check_feasibility = False  # Whether to check terrain validity and feasibility (used to check generated terrains)

        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05
        horizontal_scale = 0.05        # [m] Granularity of the grid cells in terrain plane (x, y) (Note: this influences computation time by a lot)
                                       #     When training with depth, consider using 0.1 to reduce computation time
        vertical_scale = 0.005         # [m] Granularity of the cells' height in terrain plane (z)
        border_size = 5                # [m] Size of the flat border around the entire terrain
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = True

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0.0

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 2 # starting curriculum state
        terrain_length = 18
        terrain_width = 4
        num_rows = 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 40 # number of terrain cols (types)
        
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.0,
                        "gaps": 0., 
                        "smooth flat": 0,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.2,
                        "parkour_hurdle": 0.2,
                        "parkour_flat": 0.2,
                        "parkour_step": 0.2,
                        "parkour_gap": 0.2,
                        "demo": 0.0,}
        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True

        num_goals = 8

    class commands:
        curriculum = False
        max_curriculum = 1.
        commands = [
            "lin_vel_x",    # Forward velocity
            # "lin_vel_y",    # Lateral velocity
            # "ang_vel_yaw",  # Yaw velocity
            # "heading"       # Heading (if enabled, overwrites ang_vel_yaw with heading error)
        ]
        resampling_time = 6 # time before command are changed[s]
        
        lin_vel_clip = 0.2
        ang_vel_clip = 0.4

        class ranges:
            # lin_vel_x = [0.0, 0.8]
            lin_vel_x = [0.3, 0.8]

            # Unused old values, kept for reference
            # lin_vel_x = [0.0, 2.0]  # For pre-training walking policy on flat terrain
            # lin_vel_x = [0.3, 1.2]  # For training policy on parkour terrain

            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [0.0, 0.0]

        class curriculum_ranges:
            lin_vel_x = [0.0, 1.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

        class curriculum_incremnt:
            lin_vel_x = 0.1
            lin_vel_y = 0.1
            ang_vel_yaw = 0.1
            heading = 0.5

        waypoint_delta = 0.7

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Used for original URDF
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        # Original values from Extreme Parkour
        randomize_friction = True
        friction_range = [0.6, 2.]
        randomize_base_mass = True
        added_mass_range = [0., 3.]
        randomize_base_com = True
        added_com_range = [-0.2, 0.2]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5
        randomize_motor = True
        motor_strength_range = [0.8, 1.2]

        # Inspired by Robot Parkour Learning
        # randomize_friction = True
        # friction_range = [0.2, 2.]
        # randomize_base_mass = True
        # added_mass_range = [-1.0, 1.0]
        # randomize_base_com = True
        # added_com_range = [-0.2, 0.2]
        # push_robots = True
        # push_interval_s = 8
        # max_push_vel_xy = 0.5
        # randomize_motor = True
        # motor_strength_range = [0.9, 1.1]
        
        action_delay = False                          # Enable action delay
        delay_update_global_steps = 24 * 5000         # How many steps until we update delay to the next element in action_delay_steps
        # action_delay_steps = [0.5]                      # Action delays (in steps) to apply, training starts with first element and moves to the next every delay_update_global_steps
        # action_delay_steps = [1]                      # Action delays (in steps) to apply, training starts with first element and moves to the next every delay_update_global_steps
        # action_delay_steps = [1.25]                      # Action delays (in steps) to apply, training starts with first element and moves to the next every delay_update_global_steps
        action_delay_steps = [1.5]                      # Action delays (in steps) to apply, training starts with first element and moves to the next every delay_update_global_steps
        # action_delay_steps = [2]                      # Action delays (in steps) to apply, training starts with first element and moves to the next every delay_update_global_steps
        action_buf_len = int(max(action_delay_steps) * 4) + 1  # (Multiply by 4 to account for decimation, since buffer is updated during compute_torques())
        
    class rewards:
        class scales:
            # tracking rewards
            # reach_goal = 1.5
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            # regularization rewards
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.
            dof_acc = -2.5e-7
            collision = -10.
            action_rate = -0.1
            delta_torques = -1.0e-7
            torques = -0.00001
            hip_pos = -0.5
            dof_error = -0.04
            feet_stumble = -1
            feet_edge = -1
            
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1
        soft_torque_limit = 0.4
        base_height_target = 1.
        max_contact_force = 40. # forces above this value are penalized

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
 
    class policy:
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [64, 20]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
    
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]
    
    class depth_encoder:
        if_depth = LeggedRobotCfg.depth.use_camera
        train_direction_distillation = False
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = LeggedRobotCfg.depth.update_interval * 24

    class estimator:
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = LeggedRobotCfg.env.n_priv
        num_prop = LeggedRobotCfg.env.n_proprio
        num_scan = LeggedRobotCfg.env.n_scan

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 5000 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'rough_a1'
        run_name = ''
        # load and resume
        resume = False
        load_run = None # None = exptid
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt