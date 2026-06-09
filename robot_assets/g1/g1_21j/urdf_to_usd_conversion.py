# Copyright 2026 Zachary Olkin. All rights reserved.

# Isaac Sim app must be launched FIRST
from isaaclab.app import AppLauncher

# Launch with headless mode (no GUI needed for conversion)
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

# 1. Create the configuration
cfg = UrdfConverterCfg(
    asset_path="/home/zolkin/AmberLab/Project-Isaac-RL/robot-rl/robot_rl/robot_assets/g1/g1_21j/g1_21j_merged.urdf",  # Absolute path to your URDF
    usd_dir="/home/zolkin/AmberLab/Project-Isaac-RL/robot-rl/robot_rl/robot_assets/g1/g1_21j",      # Where the USD will be saved
    usd_file_name="g1_21j_default.usd",             # Output file name (optional)
    fix_base=False,                            # Fix the base link in place
    merge_fixed_joints=False,                  # Merge links connected by fixed joints
    make_instanceable=True,                   # Memory-efficient instancing
    joint_drive=UrdfConverterCfg.JointDriveCfg(     # Dummy default values for now
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=400.0,
            damping=40.0,
        ),
    ),
)

# 2. Run the converter
converter = UrdfConverter(cfg)

# 3. Access the output path
print(f"USD saved to: {converter.usd_path}")