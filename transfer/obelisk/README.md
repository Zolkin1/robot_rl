# Running the Docker

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

Then finally we can run the stack:
```
obk-launch config_file_path=$ROBOT_RL_ROOT/g1_control/configs/rl_policy_config.yaml device_name=onboard bag=false
```

## Setting up the Xbox remote
You can make sure that you can see the controller with
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