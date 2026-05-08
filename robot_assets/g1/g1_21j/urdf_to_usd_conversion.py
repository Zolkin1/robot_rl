"""Convert G1 21-joint URDF to USD using the Isaac Lab v3 URDF importer.

Usage:
    python urdf_to_usd_conversion.py

Notes:
    - The v3 importer auto-generates output at {usd_dir}/{robot_name}/{robot_name}.usda
      For g1_21j_merged.urdf this will be: g1_21j/g1_21j_merged/g1_21j_merged.usda
    - merge_fixed_joints is handled as a URDF XML pre-processing step
    - self_collision is a direct config parameter
    - For floating-base robots (fix_base=False), ArticulationRootAPI placement is handled
      by the importer's asset transformer pipeline
    - After running, update the usd_path in assets/robots/g1_21j.py to point to the new output
"""

import argparse
import os

# Isaac Sim app must be launched before importing converter
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert G1 URDF to USD.")
# parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

cfg = UrdfConverterCfg(
    asset_path=os.path.join(SCRIPT_DIR, "g1_21j_box_merged.urdf"),
    usd_dir=SCRIPT_DIR,
    fix_base=False,
    merge_fixed_joints=False,
    make_instanceable=False,
    self_collision=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=400.0,
            damping=40.0,
        ),
    ),
)

converter = UrdfConverter(cfg)
print(f"USD saved to: {converter.usd_path}")

# Post-process: remove the invalid root_joint from all USD payload files.
# The importer creates a FixedJoint "root_joint" with body0 pointing to the root
# Xform (not a rigid body), which causes NaN in PhysX for floating-base robots.
if not cfg.fix_base:
    import pathlib
    import re

    pattern = re.compile(
        r'\n\s*(?:def PhysicsFixedJoint|over)\s+"root_joint"[^{]*\{[^}]*\}',
    )
    for usda in pathlib.Path(converter.usd_path).parent.rglob("payloads/**/*.usda"):
        text = usda.read_text()
        new_text = pattern.sub("", text)
        if new_text != text:
            usda.write_text(new_text)
            print(f"Removed root_joint from {usda}")

simulation_app.close()
