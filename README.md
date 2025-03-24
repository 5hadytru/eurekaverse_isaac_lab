# Installation
The following instructions will install everything under one Conda environment. We have tested on Ubuntu 20.04.

1. Create a new Conda environment with:
    ```
    conda create -n eurekaverse python=3.8
    conda activate eurekaverse
    ```

2. Install Eurekaverse:
    ```
    cd eurekaverse
    pip install -e . --extra-index-url=https://pypi.nvidia.com
    ```

3. Install `legged_gym` and `rsl_rl`, base code used for quadruped reinforcement learning in simulation, extended from [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour):
    ```
    pip install -e extreme-parkour/legged_gym
    pip install -e extreme-parkour/rsl_rl
    ```

4. Install Isaac Lab:
    ```
    git clone git@github.com:isaac-sim/IsaacLab.git
    sudo apt install cmake build-essential
    ./isaaclab.sh --install # or "./isaaclab.sh -i"
    ```

# Usage
1. First, set your OpenAI API key via:
    ```
    export OPENAI_API_KEY=<YOUR_KEY>
    ```

2. Now, we are ready to begin environment curriculum generation. Review the configuration in `eurekaverse/eurekaverse/config/config.yaml`. The current parameters were used for our experiments. To run generation:
    ```
    cd eurekaverse
    python run_eurekaverse.py
    ```
    The outputs will be saved in `eurekaverse/outputs/run_eurekaverse/<RUN_ID>`.

3. Afterwards, distill the final policy via:
    ```
    python distill_eurekaverse.py <YOUR_RUN_ID>
    ```
    Similarly, the outputs will be saved in `eurekaverse/outputs/distill_eurekaverse/<RUN_ID>`.

# Deployment
Our deployment infrastructure on the Unitree Go1 uses LCM for low-level commands and Docker to run the policy. Note that our Docker is only tested on the Jetson Xavier NX on the Go1. Our setup is loosely based on [LucidSim](https://github.com/lucidsim/lucidsim) and [Walk These Ways](https://github.com/Improbable-AI/walk-these-ways).

## Initial Setup

1. Connect a Realsense D435 to the middle USB port on the Go1. 3D print and mount the Realsense using [this design](https://github.com/ZiwenZhuang/parkour/blob/main/go1_ckpts/go1_camMount_30Down.step) from [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour).

2. Start up the Go1 and connect to it on your machine via Ethernet. Make sure you can ssh onto the NX (192.168.123.15).

3. Put the robot into damping mode with the controller: L2+A, L2+B, L1+L2+START. The robot should be lying on the ground afterwards.

4. Build the Docker image with:
    ```
    cd go1_deploy/docker
    docker buildx build --platform linux/arm64 -t go1-deploy:latest . --load
    ```

5. Save the image:
    ```
    docker save go1-deploy -o go1_deploy.tar
    ```

6. Copy the Docker and other necessary files over to the Go1:
    ```
    ./send_to_unitree.sh
    scp go1_deploy.tar go1-nx:/home/unitree/go1_gym/go1_gym_deploy/scripts
    ```

6. Connect onto the Go1 NX, then load the Docker:
    ```
    sudo docker load -i go1_deploy.tar
    ```

## Running the Policy

1. Connect onto the Go1 NX. You should see `eurekaverse` in the home directory (from `./send_to_unitree.sh`).

2. Start LCM:
    ```
    cd eurekaverse/go1_deploy/launch
    ./start_lcm.sh
    ```

3. Start and enter the Docker container:
    ```
    sudo -E ./start_docker.sh
    ./enter_docker.sh
    ```

4. Within the container, run the policy:
    ```
    python3 deploy_policy.py
    ```

5. Monitor the output, and when it's ready to calibrate, press R2. Pressing R2 again will start the policy, and R2 again will stop.
