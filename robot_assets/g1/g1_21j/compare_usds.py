# Copyright 2026 Zachary Olkin. All rights reserved.

"""Compare two USD files side-by-side.

Prints joint properties (stiffness, damping, limits) and body properties
(mass, inertia) for each USD, highlighting differences.

Usage:
    python compare_usds.py <usd_path_1> <usd_path_2>
"""

import argparse
from collections import OrderedDict
from typing import Any

# Isaac Sim app must be launched FIRST
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402


def extract_joint_data(stage: Usd.Stage) -> OrderedDict[str, dict[str, Any]]:
    """Extract joint properties from all joints in the USD stage.

    Returns an OrderedDict preserving traversal order, mapping joint name to
    a dict with keys: type, stiffness, damping, lower_limit, upper_limit.
    """
    joints = OrderedDict()
    root = stage.GetPseudoRoot()

    for prim in Usd.PrimRange(root):
        is_revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)

        if not (is_revolute or is_prismatic):
            continue

        joint_name = prim.GetName()
        drive_type = "angular" if is_revolute else "linear"
        joint_type = "revolute" if is_revolute else "prismatic"

        # Drive API for stiffness/damping
        drive_api = UsdPhysics.DriveAPI(prim, drive_type)
        stiffness_attr = drive_api.GetStiffnessAttr()
        damping_attr = drive_api.GetDampingAttr()

        # Joint limits
        if is_revolute:
            joint_api = UsdPhysics.RevoluteJoint(prim)
        else:
            joint_api = UsdPhysics.PrismaticJoint(prim)
        lower_attr = joint_api.GetLowerLimitAttr()
        upper_attr = joint_api.GetUpperLimitAttr()

        joints[joint_name] = {
            "path": str(prim.GetPath()),
            "type": joint_type,
            "stiffness": stiffness_attr.Get() if stiffness_attr else None,
            "damping": damping_attr.Get() if damping_attr else None,
            "lower_limit": lower_attr.Get() if lower_attr else None,
            "upper_limit": upper_attr.Get() if upper_attr else None,
        }

    return joints


def extract_body_data(stage: Usd.Stage) -> OrderedDict[str, dict[str, Any]]:
    """Extract body (rigid body) properties from all bodies in the USD stage.

    Returns an OrderedDict preserving traversal order, mapping body name to
    a dict with keys: mass, inertia.
    """
    bodies = OrderedDict()
    root = stage.GetPseudoRoot()

    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.MassAPI):
            continue

        mass_api = UsdPhysics.MassAPI(prim)
        mass_attr = mass_api.GetMassAttr()
        inertia_attr = mass_api.GetDiagonalInertiaAttr()

        bodies[prim.GetName()] = {
            "path": str(prim.GetPath()),
            "mass": mass_attr.Get() if mass_attr else None,
            "inertia": inertia_attr.Get() if inertia_attr else None,
        }

    return bodies


def _fmt_val(val: Any) -> str:
    """Format a value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.6g}"
    return str(val)


def _match_marker(val1: Any, val2: Any) -> str:
    """Return a marker indicating whether two values match."""
    if val1 == val2:
        return ""
    return " ***"


def print_joint_comparison(
    joints1: OrderedDict[str, dict],
    joints2: OrderedDict[str, dict],
    label1: str,
    label2: str,
) -> None:
    """Print a side-by-side comparison of joint properties."""
    all_names = list(dict.fromkeys(list(joints1.keys()) + list(joints2.keys())))

    print("=" * 120)
    print("JOINT COMPARISON")
    print("=" * 120)

    for name in all_names:
        d1 = joints1.get(name)
        d2 = joints2.get(name)

        print(f"\n{'─' * 120}")
        if d1 and d2:
            print(f"  Joint: {name}")
        elif d1:
            print(f"  Joint: {name}  *** ONLY IN {label1} ***")
            d2 = {k: None for k in d1}
        else:
            print(f"  Joint: {name}  *** ONLY IN {label2} ***")
            d1 = {k: None for k in d2}

        header = f"  {'Property':<20} {label1:>25}   {label2:>25}   {'Diff?':>6}"
        print(header)
        print(f"  {'─' * 20} {'─' * 25}   {'─' * 25}   {'─' * 6}")

        for key in ["type", "stiffness", "damping", "lower_limit", "upper_limit"]:
            v1, v2 = d1.get(key), d2.get(key)
            marker = _match_marker(v1, v2)
            print(f"  {key:<20} {_fmt_val(v1):>25}   {_fmt_val(v2):>25}  {marker}")

    print()


def print_body_comparison(
    bodies1: OrderedDict[str, dict],
    bodies2: OrderedDict[str, dict],
    label1: str,
    label2: str,
) -> None:
    """Print a side-by-side comparison of body properties."""
    all_names = list(dict.fromkeys(list(bodies1.keys()) + list(bodies2.keys())))

    print("=" * 120)
    print("BODY COMPARISON")
    print("=" * 120)

    for name in all_names:
        d1 = bodies1.get(name)
        d2 = bodies2.get(name)

        print(f"\n{'─' * 120}")
        if d1 and d2:
            print(f"  Body: {name}")
        elif d1:
            print(f"  Body: {name}  *** ONLY IN {label1} ***")
            d2 = {k: None for k in d1}
        else:
            print(f"  Body: {name}  *** ONLY IN {label2} ***")
            d1 = {k: None for k in d2}

        header = f"  {'Property':<20} {label1:>35}   {label2:>35}   {'Diff?':>6}"
        print(header)
        print(f"  {'─' * 20} {'─' * 35}   {'─' * 35}   {'─' * 6}")

        for key in ["mass", "inertia"]:
            v1, v2 = d1.get(key), d2.get(key)
            marker = _match_marker(v1, v2)
            print(f"  {key:<20} {_fmt_val(v1):>35}   {_fmt_val(v2):>35}  {marker}")

    print()


def print_ordering(
    joints1: OrderedDict[str, dict],
    joints2: OrderedDict[str, dict],
    bodies1: OrderedDict[str, dict],
    bodies2: OrderedDict[str, dict],
    label1: str,
    label2: str,
) -> None:
    """Print the ordering of joints and bodies in each USD."""
    print("=" * 120)
    print("JOINT ORDERING")
    print("=" * 120)
    j1_names = list(joints1.keys())
    j2_names = list(joints2.keys())
    max_joints = max(len(j1_names), len(j2_names))

    print(f"  {'Idx':<6} {label1:<40} {label2:<40} {'Match?':>6}")
    print(f"  {'─' * 6} {'─' * 40} {'─' * 40} {'─' * 6}")
    for i in range(max_joints):
        n1 = j1_names[i] if i < len(j1_names) else "---"
        n2 = j2_names[i] if i < len(j2_names) else "---"
        marker = "" if n1 == n2 else " ***"
        print(f"  {i:<6} {n1:<40} {n2:<40} {marker}")

    print()
    print("=" * 120)
    print("BODY ORDERING")
    print("=" * 120)
    b1_names = list(bodies1.keys())
    b2_names = list(bodies2.keys())
    max_bodies = max(len(b1_names), len(b2_names))

    print(f"  {'Idx':<6} {label1:<40} {label2:<40} {'Match?':>6}")
    print(f"  {'─' * 6} {'─' * 40} {'─' * 40} {'─' * 6}")
    for i in range(max_bodies):
        n1 = b1_names[i] if i < len(b1_names) else "---"
        n2 = b2_names[i] if i < len(b2_names) else "---"
        marker = "" if n1 == n2 else " ***"
        print(f"  {i:<6} {n1:<40} {n2:<40} {marker}")

    print()


def main() -> None:
    """Compare two USD files and print side-by-side property tables."""
    parser = argparse.ArgumentParser(description="Compare two USD files side-by-side.")
    parser.add_argument("usd1", type=str, help="Path to the first USD file.")
    parser.add_argument("usd2", type=str, help="Path to the second USD file.")
    args = parser.parse_args()

    label1 = args.usd1.split("/")[-1]
    label2 = args.usd2.split("/")[-1]

    print(f"\nLoading USD 1: {args.usd1}")
    stage1 = Usd.Stage.Open(args.usd1)
    print(f"Loading USD 2: {args.usd2}")
    stage2 = Usd.Stage.Open(args.usd2)

    joints1 = extract_joint_data(stage1)
    joints2 = extract_joint_data(stage2)
    bodies1 = extract_body_data(stage1)
    bodies2 = extract_body_data(stage2)

    print(f"\nFound {len(joints1)} joints and {len(bodies1)} bodies in {label1}")
    print(f"Found {len(joints2)} joints and {len(bodies2)} bodies in {label2}")
    print()

    print_joint_comparison(joints1, joints2, label1, label2)
    print_body_comparison(bodies1, bodies2, label1, label2)
    print_ordering(joints1, joints2, bodies1, bodies2, label1, label2)

    # Force exit — simulation_app.close() hangs in headless mode
    # when no simulation loop was run.
    import sys
    sys.exit(0)


if __name__ == "__main__":
    main()
