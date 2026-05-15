"""Terrain-aware override for trajectory reference-frame snaps.

The base trajectory command keeps a per-env reference-frame pose
(``cmd.ref_poses``) and re-anchors it to the current body-frame pose at
every domain transition where the new ref frame is in contact.  On flat
ground that's fine, but on terrain-dependent skills (stairs) the foot
drifts and the snap leaves the reference offset from the actual step.

This module supplies a single helper, :func:`apply_terrain_aware_ref`,
that the base trajectory command calls in place of the raw assignment.
For envs in a registered terrain-aware skill (currently ``stair_up``),
it snaps the reference to the projected stair top under the foot and
applies the per-trajectory / per-domain offset that the reset event
already uses.  Envs not in such a skill, or envs whose terrain does not
expose a projector, fall through unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .base_trajectory_cmd import BaseTrajectoryCommand


_TERRAIN_AWARE_SKILLS = ("stair_up",)


def apply_terrain_aware_ref(
    cmd: "BaseTrajectoryCommand",
    env_indices: torch.Tensor,
    default_new_ref: torch.Tensor,
    new_domain: torch.Tensor,
) -> torch.Tensor:
    """Return the ref-pose tensor to assign for the changed envs.

    For envs whose current skill is terrain-aware (``stair_up``), the
    pose is rebuilt as ``[stair_center.x + spawn_off.x, foot.y,
    stair_center.z + spawn_off.z]`` with an identity quaternion.  All
    other envs pass through ``default_new_ref`` unchanged.

    Args:
        cmd: The trajectory command term (provides ``manager``, ``env``).
        env_indices: Global env indices whose reference frame just
            switched.  Shape ``[M]``.
        default_new_ref: The body-frame snap pose the caller would have
            written.  Shape ``[M, 7]`` as ``[x, y, z, qx, qy, qz, qw]``.
        new_domain: Domain index after the switch, aligned to
            ``env_indices``.  Shape ``[M]``.

    Returns:
        Tensor of shape ``[M, 7]`` — same shape as ``default_new_ref``.
    """
    if env_indices.numel() == 0:
        return default_new_ref

    manager = cmd.manager

    skill_name_to_idx = getattr(manager, "skill_name_to_idx", None)
    if skill_name_to_idx is None:
        return default_new_ref

    terrain_skill_ids = [
        skill_name_to_idx[name] for name in _TERRAIN_AWARE_SKILLS if name in skill_name_to_idx
    ]
    if not terrain_skill_ids:
        return default_new_ref

    if "skill_idx" not in manager.data.keys():
        return default_new_ref
    skill_idx_table = manager.data["skill_idx"]

    terrain = getattr(cmd.env.scene, "terrain", None)
    terrain_meta = getattr(terrain, "terrain_meta_data", None) or {}
    project = terrain_meta.get("project", None)
    if project is None:
        return default_new_ref

    traj_idx = manager.get_current_trajectory_indices(env_indices)  # [M]
    env_skill = skill_idx_table[traj_idx]                            # [M]

    target = torch.tensor(terrain_skill_ids, dtype=env_skill.dtype, device=env_skill.device)
    is_terrain = (env_skill.unsqueeze(-1) == target).any(dim=-1)     # [M]
    if not torch.any(is_terrain):
        return default_new_ref

    sub_traj_idx = traj_idx[is_terrain]
    sub_dom_idx = new_domain[is_terrain]

    ref_off = manager.data["ref_frame_offset"][sub_traj_idx, sub_dom_idx]            # [K, 3]
    stair_off = manager.data["origin_relative_to_stair_center"][sub_traj_idx]        # [K, 3]
    spawn_off = ref_off + stair_off

    ref_xy = default_new_ref[is_terrain, :2]                                          # [K, 2]
    stair_center = project(ref_xy)                                                    # [K, 3]

    new_pose = default_new_ref.clone()
    new_pose[is_terrain, 0] = stair_center[:, 0] + spawn_off[:, 0]
    # default_new_ref[is_terrain, 1] is already the foot's current Y — leave it.
    new_pose[is_terrain, 2] = stair_center[:, 2] + spawn_off[:, 2]
    new_pose[is_terrain, 3:6] = 0.0
    new_pose[is_terrain, 6] = 1.0
    return new_pose
