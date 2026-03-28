# Running the Docker
For now, we suggest bringing the docker up in a `devcontainer` within VSCode.

## Prerequisites
- Docker and Docker Compose plugin installed
  - Installation guide: [Docker Compose Installation](https://docs.docker.com/compose/install/linux/#install-using-the-repository)
- VSCode with devcontainer support (recommended)

## Environment Setup
Set the `ROBOT_RL_ROOT` environment variable:
```bash
export ROBOT_RL_ROOT=/your/path/to/robot_rl/transfer/obelisk
```

## Creating a `.env`
In the `docker` folder you must make a `.env` file that looks like:
```
USER=${USER:-$(id -un)}
UID=1000
GID=1000
```
But you may need to switch the numbers depending on your user.

## Running in Docker
We recommend using VSCode's devcontainer feature to run the Docker environment.

### Initial Setup
1. Navigate to the obelisk folder
2. Open VSCode command palette (Ctrl+Shift+P)
3. Select "Dev Container: Rebuild and Reopen in Container"
4. Choose your preferred configuration:
   - GPU-based (recommended for better performance)
   - No-GPU (if GPU is not available)

## Pushing and Pulling from a Registry - DOES NOT WORK RIGHT NOW
Login to your registry of choice (for now I am using github packages).
(i.e. `docker login ghcr.io -u YOUR_GITHUB_USERNAME`)

To push:
```
docker compose -f <compose_file>.yml build
```
then
```
docker compose -f <compose_file>.yml push
```

To pull (i.e. on the on-board computer):
```
docker pull ghcr.io/github_username/robot-rl-obk:no-gpu-latest
```

```
docker compose -f <compose_file>.yaml up
```

# Setup within the Docker
To build the local ROS workspace you can run
```
source scripts/build.sh
```
or do it manually:
Run
```
obk
```
To build and source ROS (per terminal).

After Obelisk has been built you can just run
```
obk-build
```
and
```
obk-clean
```
to clean the Obelisk build folder.

After `obk` we can build this package:
```
colcon build --symlink-install --parallel-workers $(nproc)
```
Then we can source this package:
```
source install/setup.bash
```

Then finally we can run the stack in sim:
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/configs/sim_config_baseline.yaml device=onboard bag=false
```

## Different Configs/Policies
LIP:
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/configs/hardware_config_clf.yaml device=onboard bag=false
```

HZD:
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/configs/hardware_config_hzd_gl.yaml device=onboard bag=false
```

Baseline:
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/configs/hardware_config.yaml device=onboard bag=false
```

Running:
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/configs/hardware_config_running.yaml device=onboard bag=false
```

Bow Forward:
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/configs/bow_forward/bow_forward_config.yaml device=onboard bag=false
```

Multi-behavior:
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/configs/multi-behavior/sim_multi-behavior_config.yaml device=onboard bag=false
```

CBF-Sim
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/configs/running/cbf_sim_running_config.yaml device=onboard bag=false
```

<!-- HZD with optitrack logging:
```
obk-launch config_file_path=$ROBOT_RL_ROOT/g1_control/configs/hardware_config_hzd_gl_optitrack.yaml device_name=onboard bag=false
``` -->

# Run the Lidar
May need to run
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

```
source /home/${USER}/lidar_odom_ws/install/setup.bash
```
```
ros2 launch lidar_scan fastlio_launch.py 
```

# Running on hardware
Follow the above steps to make sure everything work in simulation.

When connected to the robot, attempt to ping the IP address at `192.168.123.161`.
Then, update the `network_interface_name` parameter in the robot section of `hardware_config.yaml` to match the name of your network interface (can be seen via `ifconfig`).

Now we can launch the hardware stack:
```
obk-launch config_file_path=$ROBOT_RL_ROOT/g1_control/configs/hardware_config.yaml device_name=onboard bag=false
```

The robot interface in Obelisk has a statemachine that we need to "navigate" to enter into low level control mode.
For the G1, we will follow this diagram (can get to L1 by holding down Left Bumper, see Obelisk docs for more info):

```
    Left Trigger        L1 DPAD Left       L1 DPAD down
init ----------> damping -------> user_pose ---------> low_level_ctrl
```
At `user_pose`, the robot will snap to the default position specified by `user_pose` in `hardware_config.yaml` and hold that.

At `low_level_ctrl`, the output from the controller node will be applied to robot.

E-STOP is right trigger.


A: Activate policy
Right Bumper: Zero
Right Dpad: integrated joystick
Left Dpad: MPC
Top Dpad: Normal Joystick
Right Bumper + Y/A: Increment/decrement speed



# Setting up the Xbox remote
You can now run the script:
```
bash scripts/joystick.sh
```

Or you can do it manually:
You can make sure that you can see the remote control with
```
sudo evtest
```

Then you can run
```
sudo chmod 666 /dev/input/eventX
```
where `X` is the number that you saw from evtest.

Then we can verify that ROS2 can see it with:
```
ros2 run joy joy_enumerate_devices
```


# Running the docker through tmux session (detached)
#TODO: check if we need --pull always

Start a tmux session
```tmux```

Check existing tmux sessions
```tmux ls```

To dettach from a tmux session
```ctrl +b then d```

To reattach to a tmux session
```tmux attach -t <id>```

If you want to build the container with no cache:
```docker compose -f docker-compose-no-gpu.yml build --no-cache```

To create and start the container, run the following command:
```docker compose -f docker-compose-no-gpu.yml up ```
Note this would also build the container if necessary.

To check existing docker images
```docker ps```

To check running containers
```docker container ls```

To open a new bash terminal for running container
```docker exec -it <container_id or container_name> /bin/bash```

# Plotting Odomety
To plot the odometry run
```
python plot_odom.py
```
or 
```
python plot_odom.py /path/to/odom_data.csv
```

You can also use `--start-time` and `--end-time` to specify and start and end time.

# Plotting the Observations and Actions
To plot these, run:
```
python plot_ctrl.py
```

You can also use `--start-time` and `--end-time` to specify and start and end time.

## Setting the IP setting through the terminal

Check the names:
```
nmcli con show
```

To set it to manual:
```
nmcli con mod "Wired connection 1" ipv4.addresses 192.168.123.222/24 ipv4.method manual
nmcli con up "Wired connection 1"
```

To set to automatic:
```
nmcli con mod "Wired connection 1" ipv4.method auto ipv4.addresses ""
nmcli con up "Wired connection 1"
```

# Using Optitrack
We are using the natnet driver located [here](https://github.com/L2S-lab/natnet_ros2). In the docker it is installed to /home/{USER}.

Once you are inside the docker you need to build the package using colcon, but in order to do that you need to activate obelisk so you have ros2. So run:

```
obk
```

make sure you are in `~/natnet_ros2` then run

```
colcon build --symlink-install
. install/setup.bash
```

to build it.

Run it with:
```
ros2 launch natnet_ros2 gui_natnet_ros2.launch.py
```

You may need to run
```
mkdir -p /tmp/runtime-$USER
chmod 700 /tmp/runtime-$USER
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
```

but I don't know for sure.

To run the optitrack forwarding node:
```
ros2 run g1_control optitrack_forwarding
```