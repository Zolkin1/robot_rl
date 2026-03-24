# Robot RL
## :construction: This repo is under active development :construction:
Since we actively develop on this repo we may push major/breaking changes.
We do not guarantee any kind of API/environment stability right now, although this could come in the future.
That being said, we attempt to keep the repo as usable and documented as possible given its developmental status.

## Overview

This project is a set of tools for end-to-end development of RL for robots. Specifically, we support:

- RL development using IsaacLab and IsaacSim.
- sim2sim transfer using Mujoco.
- Hardware transfer using Obelisk (ROS2) - note that ROS2 is NOT a dependency of this project - the hardware
interface can be run through a docker/dev-container provided in this repo. See below for more information.

## Installation
When you clone this repo, please use **Git Large File System (lfs)**.

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/robot_rl

- Verify that the extension is correctly by attempting to train:

    - Running a task (see below for a full list of tasks):

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

## Running Tasks
To train a policy run:
```bash
python scripts/rsl_rl/train_policy.py --env_type=<ENV_NAME> --headless
```

To add a run name add `--run_name=my_run_name`. This will add the name after the date on the folder with the run name.

Note that right now the only RL library that is tested in `RSL_RL`.

To play the most recently trained policy for a given task run:
```bash
python scripts/rsl_rl/play_policy.py --env_type=<ENV_NAME> --log_data --export_policy --headless
```

for a specific run you can pass in additional config such as `--load_run=<run_dir>`
If you want to play from a specific checkpoint then you can run the play script with `--checkpoint=<checkpoint>` (`<checkpoint>` is something like "model_1800").

For both `train` and `play` you can also specify a number of envs with `--num_envs=###`.

### Preparing for Hardware/Uploading to Hugging Face
We suggest uploading the policy to hugging face when you are ready to run it on hardware so that this repo (the code repo) is kept clean.
You can upload to hugging face automatically with
```bash
python scripts/rsl_rl/export_to_hardware.py --env_type=<env-type> --load_run=<run_dir> --hf_repo_id=<username/repo> --policy_name=<policy_name>
```
This will load a already exported run and upload the exported policy to hugging face in the specified repo with the specified name.

NOTE: The policy must already be exported (see `play_policy` above).

For now we are uploading the policies to the hugging face repo [here](https://huggingface.co/zolkin/robot_rl/tree/main).

These policies are automatically downloaded in the `transfer/obelisk` controller.
## RL Tasks

RL Task list:

| Env. Type             |   Robot    |      Hardware Tested?      | Description                                                                          |
|-----------------------|:----------:|:--------------------------:|--------------------------------------------------------------------------------------|
| `vanilla`             |     G1     |     :white_check_mark:     | Basic, hand-tuned, RL walking on the G1 humanoid on flat ground.                     |
| `lip_clf`             |     G1     |     :white_check_mark:     | Basic, LIP CLF RL walking on the G1 humanoid on flat ground.                         |
| `walking_clf`         |     G1     |             No             | A HZD gait library; CLF RL walking on the G1 humanoid on flat ground.                |
| `walking_clf_sym`     |     G1     |     :white_check_mark:     | A HZD gait library; CLF RL walking on the G1 humanoid on flat ground, uses symmetry. |
| `running_clf_sym`     | G1 |     :white_check_mark:     | Running controller using a gait library.                                             |
| `waving_clf`          | G1 |             No             | Robot stands still and waves. Uses trajectories and CLFs.                            |
| `bow_forward_clf_sym` | G1 |     :white_check_mark:     | Robot bows forward then holds its position. Uses symmetry.                           |
| `bend_up_clf_sym`     | G1 |     :white_check_mark:     | Robot bends back up then holds in the standing position. Uses symmetry.              |

You can also append "_ec" to any of the above tasks to run them on the robot model that has the additional weight from the extra compute (EC).

## Copying checkpoints from remote server 
First mount the server to your local desktop
 
```
bash scripts/mount_remote.sh
```

Note: this script might require a few changes as somethings are hard-coded.

## sim2sim Transfer
This code base has a built in sim2sim transfer (i.e. the policy is trained in IsaacLab and can be run in Mujoco).
Currently, we only support the G1 (as that is the only policy we have right now), but the code is easily extended to other robots.
To run the sim2sim transfer, go to the `transfer/sim/` directory. From this directory run
```
python g1_runner.py --env_type=<ENV_NAME>
```
where `ENV_NAME` is the same as above: it is an entry into the table. The arguments here follow the same structure as the `play_policy.py` script.
Optionally you can also add `--load_run=<run_dir>` to run a specific policy, otherwise the newest one will be run.
`--log` enables logging to the same directory. You can choose the logging directory with `--log_dir`.

This uses information that must have been exported from the `play_policy.py` script.

Then we can generate plots for the mujoco simulation. We can run
```
python plot_from_sim.py --env_type=<ENV_NAME>
```
and optionally you can pass `--load_run=<run_dir>` where `<run_dir>` is something like `025-11-17_15-04-12_walking_test`.
You can also pass `--log_session=<folder>` to choose a folder from within those mujoco logs. 
If either of these are unspecified then we plot the data for the newest run.

To add a new robot, the associated robot sim files will need to be added into the `transfer/sim/robots/` folder,
the `rl_policy_wrapper` will need to be adjusted a bit, and a new `runner` file will need to be made.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Obelisk Transfer
First, set the environment variable `ROBOT_RL_ROOT` to the path to the `/transfer/obelisk` folder.
Now we can being building the docker container.

### Prerequisites
- Docker and Docker Compose plugin installed
  - Installation guide: [Docker Compose Installation](https://docs.docker.com/compose/install/linux/#install-using-the-repository)
- VSCode with devcontainer support (recommended)

We recommend using VSCode's devcontainer feature to run the Docker environment, although in theory you can run this
as a vanilla docker container.

### Initial Setup (VSCode devcontainer)
1. Navigate to the Obelisk folder in VSCode.
2. Open VSCode command palette (`Ctrl+Shift+P`)
3. Select "Dev Container: Rebuild and Reopen in Container"
4. Choose your preferred configuration:
   - GPU-based (recommended for better performance)
   - No-GPU (if GPU is not available)

At anytime you can open the folder locally by using `Ctrl+Shift+P` then "Dev Container: Open Folder Locally".

At this point you are now inside the docker container and can now use Obelisk and ROS2. Please see the readme in the
Obelisk folder for further instructions.


## Other Dependencies
To run the sim2sim transfer, you will to install these dependencies in your conda environment:
- `pygame`
- `mujoco`
- `huggingface_hub`

## Updating IsaacLab
Sometimes you will want to updated the version of IsaacLab you are using. To do this, go to the IsaacLab directory
(where you cloned it). Then pull the version you want from git.
Then in that folder run `./isaaclab.sh --install`.


## Citing
If you use this repo in any publications, please consider citing the relevant paper (see below).

For the original CLF-RL idea and basic walking:
```
@article{li_clf-rl_2026,
  author={Li, Kejun and Olkin, Zachary and Yue, Yisong and Ames, Aaron D.},
  journal={IEEE Robotics and Automation Letters}, 
  title={CLF-RL: Control Lyapunov Function Guided Reinforcement Learning}, 
  year={2026},
  volume={11},
  number={3},
  pages={3230-3237},
  doi={10.1109/LRA.2026.3653329}}
```
For the running:
```
@misc{olkin_chasing_2025,
      title={Chasing Stability: Humanoid Running via Control Lyapunov Function Guided Reinforcement Learning}, 
      author={Zachary Olkin and Kejun Li and William D. Compton and Aaron D. Ames},
      year={2025},
      eprint={2509.19573},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.19573}, 
}
```
For the stepping stones:
```
@misc{dai_walk_2026,
      title={Walk the PLANC: Physics-Guided RL for Agile Humanoid Locomotion on Constrained Footholds}, 
      author={Min Dai and William D. Compton and Junheng Li and Lizhi Yang and Aaron D. Ames},
      year={2026},
      eprint={2601.06286},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.06286}, 
}
```