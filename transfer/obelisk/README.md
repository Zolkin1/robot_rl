# Running the Docker
For now, we suggest bringing the docker up in a `devcontainer` within vscode.

# Setup within the Docker
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
obk-launch config_file_path=$ROBOT_RL_ROOT/g1_control/configs/rl_policy_config.yaml device_name=onboard bag=false
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
For the G1, we will follow this diagram:

```
     dpad right          squares            dpad down
init ----------> damping -------> user_pose ---------> low_level_ctrl
```
At `user_pose`, the robot will snap to the default position specified by `user_pose` in `hardware_config.yaml` and hold that.

At low level ctrl, the output from the controller node will be applied to robot.

TODO: Still need to test the hardware config on the hardware.

# Setting up the Xbox remote
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