"""Helpers for cross-fading trajectory outputs across a skill change.

When the multi-skill command commits a new skill at a contact-gate fire it
starts a transition: for ``blend_end_phi`` worth of phi after the commit the
desired output is a convex blend of the *old* skill's trajectory and the
*new* skill's trajectory, both sampled at the same per-env phase.  Linear
blend is used for positions / joint angles / velocities; the orientation
quaternions per body are SLERPed instead so the result stays on the unit
sphere and follows the shortest rotation.
"""

from __future__ import annotations

import torch


def quat_slerp(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Vectorised spherical linear interpolation between unit quaternions.

    Args:
        q1: ``[..., 4]`` start quaternion in ``(x, y, z, w)`` order.
        q2: ``[..., 4]`` end quaternion in ``(x, y, z, w)`` order.
        t: Interpolation parameter in ``[0, 1]``, broadcastable to ``q1``'s
            leading dims (i.e. ``[...]``).

    Returns:
        ``[..., 4]`` SLERP-interpolated quaternion, re-normalised to unit
        length.  Near-parallel inputs fall back to a normalised lerp to
        avoid the ``1 / sin(0)`` blowup.
    """
    dot = (q1 * q2).sum(-1, keepdim=True)
    # Shortest-path: flip q2 if dot < 0 (q and -q represent the same rotation).
    q2 = torch.where(dot < 0, -q2, q2)
    dot = dot.abs().clamp(max=1.0 - 1e-6)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    t_ = t.unsqueeze(-1)

    w1 = torch.sin((1.0 - t_) * theta) / sin_theta
    w2 = torch.sin(t_ * theta) / sin_theta
    out = w1 * q1 + w2 * q2

    # Near-parallel: lerp + normalise (sin(theta) ~ 0 path is numerically bad).
    near = (1.0 - dot).squeeze(-1) < 1e-6
    lerp = (1.0 - t_) * q1 + t_ * q2
    out = torch.where(near.unsqueeze(-1), lerp, out)
    return torch.nn.functional.normalize(out, dim=-1)


def blend_outputs(
    y_old: torch.Tensor,
    y_new: torch.Tensor,
    alpha: torch.Tensor,
    quat_index_groups: torch.Tensor | None,
) -> torch.Tensor:
    """Blend two trajectory output tensors with per-env weight ``alpha``.

    Args:
        y_old: ``[M, P]`` outputs from the fading-out trajectory.
        y_new: ``[M, P]`` outputs from the fading-in trajectory (same column
            layout as ``y_old``).
        alpha: ``[M]`` blend weight in ``[0, 1]``; 0 → pure ``y_old``,
            1 → pure ``y_new``.
        quat_index_groups: Optional ``[G, 4]`` long tensor where row ``g``
            holds the column indices of the ``(ori_x, ori_y, ori_z, ori_w)``
            of one body in the output layout.  Those 4-column groups are
            SLERPed instead of linearly blended.  Pass ``None`` if the
            output has no orientation quaternions.

    Returns:
        ``[M, P]`` blended output.  Non-quaternion columns are
        ``(1 - alpha) * y_old + alpha * y_new``.
    """
    a = alpha.unsqueeze(-1)
    blended = (1.0 - a) * y_old + a * y_new

    if quat_index_groups is None or quat_index_groups.numel() == 0:
        return blended

    # Gather quaternions: [M, G, 4].
    q_old = y_old[:, quat_index_groups]
    q_new = y_new[:, quat_index_groups]
    # alpha broadcasts across G: [M, 1] → [M, G].
    alpha_g = alpha.unsqueeze(-1).expand(-1, quat_index_groups.shape[0])
    q_slerp = quat_slerp(q_old, q_new, alpha_g)                # [M, G, 4]

    # Scatter back into the blended tensor.
    blended[:, quat_index_groups] = q_slerp
    return blended
