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


def stair_stance_ref(
    manager,
    project,
    traj_idx: torch.Tensor,
    foot_xyz: torch.Tensor,
    domain: torch.Tensor,
) -> torch.Tensor:
    """Snapped world stance-reference position for terrain-aware trajectories.

    Pure geometry helper shared by :func:`apply_terrain_aware_ref` (which
    re-anchors ``cmd.ref_poses`` at each stair domain switch) and the
    terrain-approach planner (which predicts where a stair trajectory's
    first stance foot would snap to, to decide whether a slow-down step is
    needed).  Mirrors the composition used at episode reset
    (``reset_on_reference``): ``project(foot) + ref_frame_offset[traj,dom]
    + origin_relative_to_stair_center[traj]`` in x/z, with y left at the
    foot's current value.

    Args:
        manager: The :class:`MultiSkillManager` (provides the per-traj /
            per-domain offset tables in ``manager.data``).
        project: The terrain projector callable
            (``terrain.terrain_meta_data["project"]``) mapping a foot
            ``[K, 3]`` world xyz to its snapped stair tread / entry xyz.
        traj_idx: ``[K]`` global stair-trajectory indices.
        foot_xyz: ``[K, 3]`` world foot positions to project.  The z is
            used by ``project`` to disambiguate which tread the foot is on.
        domain: ``[K]`` domain index per row (selects the per-domain
            ``ref_frame_offset``).

    Returns:
        ``[K, 3]`` world xyz of the snapped stance reference.
    """
    stair_center = project(foot_xyz)                                       # [K, 3]
    ref_off = manager.data["ref_frame_offset"][traj_idx, domain]           # [K, 3]
    stair_off = manager.data["origin_relative_to_stair_center"][traj_idx]  # [K, 3]
    spawn_off = ref_off + stair_off
    out = foot_xyz.clone()
    out[:, 0] = stair_center[:, 0] + spawn_off[:, 0]
    # y stays at the foot's current value (matches apply_terrain_aware_ref).
    out[:, 2] = stair_center[:, 2] + spawn_off[:, 2]
    return out


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

    # Pass the foot's full xyz: the projector uses the height to pick the
    # tread the foot is physically on, so a foot whose toe is up on a step
    # while the ankle's xy still sits over the flat below the riser snaps
    # to the correct (upper) step rather than a whole step too low.
    ref_xyz = default_new_ref[is_terrain, :3]                                         # [K, 3]
    snapped = stair_stance_ref(manager, project, sub_traj_idx, ref_xyz, sub_dom_idx)  # [K, 3]

    new_pose = default_new_ref.clone()
    # ``stair_stance_ref`` keeps the foot's current y, so the [K, 3] result
    # carries x/z snapped + y unchanged — assign all three.
    new_pose[is_terrain, :3] = snapped
    new_pose[is_terrain, 3:6] = 0.0
    new_pose[is_terrain, 6] = 1.0
    return new_pose
