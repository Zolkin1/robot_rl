# Code Overview
This code deals with Reinforcement Learning (RL) for robots, specifically humanoid robots right now.
The code holds all the information to train the RL algorithm and then deploy it in simulation and on hardware.

# Code Architecture
The code is using IsaacLab at its core and thus uses the suggested file structure. 

source is where most of the core code is housed. 
source/robot_rl/robot_rl/tasks/manager_based/robot_rl/g1 is where the configs for each of the different actions are.
source/robot_rl/robot_rl/tasks/manager_based/robot_rl/mdp is where all the mdp code is including:
    - rewards
    - curriculums
    - events
    - commands
    - terminations

scripts/rsl_rl is where the code to actually run the training and the play the result is.
Training and playing is logged to log.

transfer holds code to transfer the RL code to mujoco (in transfer/sim) and to the hardware (transfer/obelisk).

# General Guidelines
- Don't care for backwards compatibility.
- Use uv virtual environment env_isaaaclab for running all related python code.
- When writing a function, always provide a doc string.
- In general, we should always use type hints where it is idiomatic.
- In python variable names should be snake_case and function should be snake_case and classes should be UpperCamelCase.