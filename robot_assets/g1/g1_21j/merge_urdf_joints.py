#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""
URDF Fixed Joint Merger

Selectively merges fixed joints in a URDF file by combining child links into
their parent links. Properly combines mass, center of mass, and inertia tensors
using the parallel axis theorem.

Usage:
    python merge_urdf_fixed_joints.py input.urdf output.urdf --keep joint1 joint2 ...

All fixed joints NOT in the --keep list will be merged.
"""

import argparse
import copy
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np


def parse_origin(element):
    """Parse an <origin> element into translation and rotation."""
    if element is None:
        return np.zeros(3), np.zeros(3)
    xyz = np.array([float(x) for x in element.get("xyz", "0 0 0").split()])
    rpy = np.array([float(x) for x in element.get("rpy", "0 0 0").split()])
    return xyz, rpy


def rpy_to_rotation_matrix(rpy):
    """Convert roll-pitch-yaw angles to a 3x3 rotation matrix."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ],
    ])
    return R


def rotation_matrix_to_rpy(R):
    """Convert a 3x3 rotation matrix to roll-pitch-yaw angles."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw])


def inertia_from_element(inertial_elem):
    """Extract the 3x3 inertia tensor from an <inertial> element."""
    if inertial_elem is None:
        return np.zeros((3, 3))

    inertia_elem = inertial_elem.find("inertia")
    if inertia_elem is None:
        return np.zeros((3, 3))

    ixx = float(inertia_elem.get("ixx", "0"))
    ixy = float(inertia_elem.get("ixy", "0"))
    ixz = float(inertia_elem.get("ixz", "0"))
    iyy = float(inertia_elem.get("iyy", "0"))
    iyz = float(inertia_elem.get("iyz", "0"))
    izz = float(inertia_elem.get("izz", "0"))

    return np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ])


def mass_from_element(inertial_elem):
    """Extract mass from an <inertial> element."""
    if inertial_elem is None:
        return 0.0
    mass_elem = inertial_elem.find("mass")
    if mass_elem is None:
        return 0.0
    return float(mass_elem.get("value", "0"))


def com_from_element(inertial_elem):
    """Extract center of mass position from an <inertial> element."""
    if inertial_elem is None:
        return np.zeros(3)
    origin_elem = inertial_elem.find("origin")
    if origin_elem is None:
        return np.zeros(3)
    return np.array([float(x) for x in origin_elem.get("xyz", "0 0 0").split()])


def com_rpy_from_element(inertial_elem):
    """Extract center of mass orientation from an <inertial> element."""
    if inertial_elem is None:
        return np.zeros(3)
    origin_elem = inertial_elem.find("origin")
    if origin_elem is None:
        return np.zeros(3)
    return np.array([float(x) for x in origin_elem.get("rpy", "0 0 0").split()])


def parallel_axis_theorem(I_body, R_body, mass, com_in_parent):
    """
    Transform an inertia tensor to a new frame using the parallel axis theorem.

    Args:
        I_body: 3x3 inertia tensor at the body's own CoM, in body frame
        R_body: 3x3 rotation from body frame to parent frame
        mass: mass of the body
        com_in_parent: position of the body's CoM in the parent frame

    Returns:
        3x3 inertia tensor expressed at the parent frame origin, in parent frame
    """
    # Rotate inertia tensor into the parent frame
    I_rotated = R_body @ I_body @ R_body.T

    # Apply parallel axis theorem: I = I_com + m * (r^T r E - r r^T)
    r = com_in_parent
    r_sq = np.dot(r, r)
    I_shifted = I_rotated + mass * (r_sq * np.eye(3) - np.outer(r, r))

    return I_shifted


def combine_inertials(
    mass_p, com_p, rpy_p, I_p,
    mass_c, com_c_in_parent, rpy_c_in_parent, I_c,
    R_joint, t_joint
):
    """
    Combine the inertial properties of a parent link and a child link
    (connected by a fixed joint) into a single set of inertial properties
    expressed in the parent link frame.

    Args:
        mass_p: mass of parent
        com_p: CoM of parent in parent frame
        rpy_p: orientation of parent inertia frame
        I_p: 3x3 inertia of parent at its CoM, in its inertia frame
        mass_c: mass of child
        com_c_in_parent: CoM of child expressed in parent frame
        rpy_c_in_parent: orientation of child's inertia frame expressed in parent frame
        I_c: 3x3 inertia of child at its CoM, in its inertia frame
        R_joint: rotation from child frame to parent frame
        t_joint: translation from parent frame to child frame origin, in parent frame

    Returns:
        (combined_mass, combined_com, combined_I) all in parent frame
    """
    total_mass = mass_p + mass_c

    if total_mass < 1e-12:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    # Combined center of mass
    combined_com = (mass_p * com_p + mass_c * com_c_in_parent) / total_mass

    # Parent inertia at combined CoM
    R_p_inertia = rpy_to_rotation_matrix(rpy_p)
    d_p = com_p - combined_com
    I_p_at_combined = parallel_axis_theorem(I_p, R_p_inertia, mass_p, d_p)

    # Child inertia at combined CoM
    R_c_inertia = rpy_to_rotation_matrix(rpy_c_in_parent)
    d_c = com_c_in_parent - combined_com
    I_c_at_combined = parallel_axis_theorem(I_c, R_c_inertia, mass_c, d_c)

    combined_I = I_p_at_combined + I_c_at_combined

    return total_mass, combined_com, combined_I


def set_origin(element, xyz, rpy):
    """Set or create an <origin> sub-element with given xyz and rpy."""
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")

    xyz_str = " ".join(f"{v:.10g}" for v in xyz)
    rpy_str = " ".join(f"{v:.10g}" for v in rpy)
    origin.set("xyz", xyz_str)
    origin.set("rpy", rpy_str)


def set_inertial(link_elem, mass, com, I):
    """Set the <inertial> element of a link."""
    # Remove existing inertial
    existing = link_elem.find("inertial")
    if existing is not None:
        link_elem.remove(existing)

    if mass < 1e-12:
        return

    inertial = ET.SubElement(link_elem, "inertial")

    origin = ET.SubElement(inertial, "origin")
    origin.set("xyz", " ".join(f"{v:.10g}" for v in com))
    origin.set("rpy", "0 0 0")

    mass_elem = ET.SubElement(inertial, "mass")
    mass_elem.set("value", f"{mass:.10g}")

    inertia_elem = ET.SubElement(inertial, "inertia")
    inertia_elem.set("ixx", f"{I[0, 0]:.10g}")
    inertia_elem.set("ixy", f"{I[0, 1]:.10g}")
    inertia_elem.set("ixz", f"{I[0, 2]:.10g}")
    inertia_elem.set("iyy", f"{I[1, 1]:.10g}")
    inertia_elem.set("iyz", f"{I[1, 2]:.10g}")
    inertia_elem.set("izz", f"{I[2, 2]:.10g}")


def transform_sub_element(elem, R_joint, t_joint):
    """
    Transform the origin of a visual/collision element from child frame to parent frame.
    """
    origin = elem.find("origin")
    xyz, rpy = parse_origin(origin)

    # Transform position: p_parent = R_joint * p_child + t_joint
    new_xyz = R_joint @ xyz + t_joint

    # Transform orientation
    R_elem = rpy_to_rotation_matrix(rpy)
    R_new = R_joint @ R_elem
    new_rpy = rotation_matrix_to_rpy(R_new)

    if origin is None:
        origin = ET.SubElement(elem, "origin")
    origin.set("xyz", " ".join(f"{v:.10g}" for v in new_xyz))
    origin.set("rpy", " ".join(f"{v:.10g}" for v in new_rpy))


def merge_fixed_joints(urdf_path, output_path, joints_to_keep):
    """
    Merge fixed joints in a URDF, keeping specified joints intact.

    Args:
        urdf_path: Path to input URDF file
        output_path: Path to write the output URDF file
        joints_to_keep: Set of fixed joint names to NOT merge
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Build lookup tables
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}

    # Find fixed joints to merge (not in keep list)
    fixed_joints_to_merge = []
    for jname, joint in joints.items():
        if joint.get("type") == "fixed" and jname not in joints_to_keep:
            fixed_joints_to_merge.append(jname)

    print(f"Found {len(fixed_joints_to_merge)} fixed joints to merge:")
    for jn in fixed_joints_to_merge:
        j = joints[jn]
        print(f"  {jn}: {j.find('parent').get('link')} <- {j.find('child').get('link')}")

    # We need to process merges in dependency order (leaves first won't work
    # because we might have chains). Instead, process repeatedly until no more
    # merges are possible.
    merged = set()
    # Map from old link name to the link name it was merged into
    remap = {}

    def resolve(link_name):
        """Follow the remap chain to find the final parent."""
        visited = set()
        while link_name in remap:
            if link_name in visited:
                break
            visited.add(link_name)
            link_name = remap[link_name]
        return link_name

    changed = True
    while changed:
        changed = False
        for jname in fixed_joints_to_merge:
            if jname in merged:
                continue

            joint = joints[jname]
            parent_name = resolve(joint.find("parent").get("link"))
            child_name = joint.find("child").get("link")

            # Skip if child was already merged
            if child_name in remap:
                merged.add(jname)
                changed = True
                continue

            # Check that parent link still exists
            if parent_name not in links:
                continue

            parent_link = links[parent_name]
            child_link = links.get(child_name)
            if child_link is None:
                merged.add(jname)
                changed = True
                continue

            # Get the joint transform
            joint_origin = joint.find("origin")
            t_joint, rpy_joint = parse_origin(joint_origin)
            R_joint = rpy_to_rotation_matrix(rpy_joint)

            # Get parent inertial properties
            parent_inertial = parent_link.find("inertial")
            mass_p = mass_from_element(parent_inertial)
            com_p = com_from_element(parent_inertial)
            rpy_p = com_rpy_from_element(parent_inertial)
            I_p = inertia_from_element(parent_inertial)

            # Get child inertial properties
            child_inertial = child_link.find("inertial")
            mass_c = mass_from_element(child_inertial)
            com_c_local = com_from_element(child_inertial)
            rpy_c_local = com_rpy_from_element(child_inertial)
            I_c = inertia_from_element(child_inertial)

            # Transform child CoM to parent frame
            com_c_in_parent = R_joint @ com_c_local + t_joint

            # Transform child inertia orientation to parent frame
            R_c_local = rpy_to_rotation_matrix(rpy_c_local)
            R_c_in_parent = R_joint @ R_c_local
            rpy_c_in_parent = rotation_matrix_to_rpy(R_c_in_parent)

            # Combine inertials
            total_mass, combined_com, combined_I = combine_inertials(
                mass_p, com_p, rpy_p, I_p,
                mass_c, com_c_in_parent, rpy_c_in_parent, I_c,
                R_joint, t_joint,
            )

            # Set combined inertial on parent
            set_inertial(parent_link, total_mass, combined_com, combined_I)

            # Move child's visual elements to parent (with transformed origins)
            for visual in child_link.findall("visual"):
                new_visual = copy.deepcopy(visual)
                transform_sub_element(new_visual, R_joint, t_joint)
                parent_link.append(new_visual)

            # Move child's collision elements to parent (with transformed origins)
            for collision in child_link.findall("collision"):
                new_collision = copy.deepcopy(collision)
                transform_sub_element(new_collision, R_joint, t_joint)
                parent_link.append(new_collision)

            # Reparent any joints that had child_name as parent
            for other_jname, other_joint in joints.items():
                if other_jname == jname:
                    continue
                other_parent = other_joint.find("parent")
                if other_parent is not None and other_parent.get("link") == child_name:
                    # Update parent to the merged parent
                    other_parent.set("link", parent_name)

                    # Update the joint origin to account for the removed link
                    other_origin = other_joint.find("origin")
                    other_xyz, other_rpy = parse_origin(other_origin)

                    # New origin = joint_transform * other_origin
                    new_xyz = R_joint @ other_xyz + t_joint
                    R_other = rpy_to_rotation_matrix(other_rpy)
                    R_new = R_joint @ R_other
                    new_rpy = rotation_matrix_to_rpy(R_new)

                    if other_origin is None:
                        other_origin = ET.SubElement(other_joint, "origin")
                    other_origin.set("xyz", " ".join(f"{v:.10g}" for v in new_xyz))
                    other_origin.set("rpy", " ".join(f"{v:.10g}" for v in new_rpy))

            # Remove the child link and the fixed joint from the tree
            root.remove(child_link)
            root.remove(joint)
            del links[child_name]

            remap[child_name] = parent_name
            merged.add(jname)
            changed = True

            print(f"  Merged '{child_name}' into '{parent_name}' via '{jname}'")

    # Pretty print
    rough_string = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    pretty = reparsed.toprettyxml(indent="  ")

    # Remove extra XML declaration and blank lines
    lines = pretty.split("\n")
    lines = [l for l in lines if l.strip()]
    # Keep XML declaration
    if lines and lines[0].startswith("<?xml"):
        output_lines = [lines[0]]
        output_lines.extend(lines[1:])
    else:
        output_lines = lines

    with open(output_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")

    print(f"\nOutput written to: {output_path}")

    # Print summary
    remaining_links = [l.get("name") for l in root.findall("link")]
    remaining_joints = [j.get("name") for j in root.findall("joint")]
    print(f"\nRemaining links ({len(remaining_links)}):")
    for ln in remaining_links:
        print(f"  {ln}")
    print(f"\nRemaining joints ({len(remaining_joints)}):")
    for jn in remaining_joints:
        j_elem = root.find(f".//joint[@name='{jn}']")
        jtype = j_elem.get("type") if j_elem is not None else "?"
        print(f"  {jn} ({jtype})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge fixed joints in a URDF file")
    parser.add_argument("input", help="Input URDF file path")
    parser.add_argument("output", help="Output URDF file path")
    parser.add_argument(
        "--keep",
        nargs="*",
        default=[],
        help="Names of fixed joints to KEEP (not merge)",
    )
    args = parser.parse_args()

    merge_fixed_joints(args.input, args.output, set(args.keep))