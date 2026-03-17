# Isaac Sim app must be launched FIRST
from isaaclab.app import AppLauncher

# Launch with headless mode (no GUI needed for conversion)
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Create the configuration
cfg = UrdfConverterCfg(
    asset_path=os.path.join(SCRIPT_DIR, "double_integrator.urdf"),
    usd_dir=SCRIPT_DIR,
    usd_file_name="double_integrator.usd",
    fix_base=True,
    merge_fixed_joints=False,
    make_instanceable=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0.0,
            damping=0.0,
        ),
    ),
)

# 2. Run the converter
converter = UrdfConverter(cfg)

# 3. Access the output path
print(f"USD saved to: {converter.usd_path}")
