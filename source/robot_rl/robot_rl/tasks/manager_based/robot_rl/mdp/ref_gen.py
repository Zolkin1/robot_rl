import math
import torch
# Combination formula for Bezier coefficients
def _ncr(n: int, r: int) -> int:
    return math.comb(n, r)


def bezier_deg(
    order: int,
    tau: torch.Tensor,
    step_dur: torch.Tensor,
    control_points: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """
    Computes the Bezier curve (order=0) or its time-derivative (order=1).
    Args:
        order: 0 for position, 1 for derivative
        tau: Tensor of shape [batch], clipped to [0,1]
        step_dur: Tensor of shape [batch]
        control_points: Tensor of shape [batch, degree+1]
        degree: polynomial degree
    Returns:
        Tensor of shape [batch]
    """
    # Ensure tau and step_dur are [batch]
    tau = torch.clamp(tau, 0.0, 1.0)
    batch = tau.size(0)

    if order == 1:
        # derivative of Bezier
        cp_diff = control_points[:, 1:] - control_points[:, :-1]  # [batch, degree]
        coefs = torch.tensor([_ncr(degree - 1, i) for i in range(degree)],
                             dtype=control_points.dtype,
                             device=control_points.device)  # [degree]
        i = torch.arange(degree, device=control_points.device)
        tau_pow = tau.unsqueeze(1) ** i.unsqueeze(0)                # [batch, degree]
        one_minus_pow = (1 - tau).unsqueeze(1) ** (degree - 1 - i).unsqueeze(0)  # [batch, degree]
        terms = degree * cp_diff * coefs.unsqueeze(0) * one_minus_pow * tau_pow
        dB = terms.sum(dim=1) / step_dur                              # [batch]
        return dB
    else:
        # position of Bezier
        coefs = torch.tensor([_ncr(degree, i) for i in range(degree + 1)],
                             dtype=control_points.dtype,
                             device=control_points.device)  # [degree+1]
        i = torch.arange(degree + 1, device=control_points.device)
        tau_pow = tau.unsqueeze(1) ** i.unsqueeze(0)                 # [batch, degree+1]
        one_minus_pow = (1 - tau).unsqueeze(1) ** (degree - i).unsqueeze(0)  # [batch, degree+1]
        terms = control_points * coefs.unsqueeze(0) * one_minus_pow * tau_pow  # [batch, degree+1]
        B = terms.sum(dim=1)                                          # [batch]
        return B


def calculate_cur_swing_foot_pos(
    bht: torch.Tensor,
    z_init: torch.Tensor,
    z_sw_max: torch.Tensor,
    tau: torch.Tensor,
    T_gait: torch.Tensor,
    zsw_neg: torch.Tensor,
    clipped_step_x: torch.Tensor,
    clipped_step_y: torch.Tensor,
) -> torch.Tensor:
    """
    Batch-friendly swing foot position calculation.
    Args:
        bht: [batch]
        p_sw0: [batch,3]
        z_sw_max: [batch]
        tau: [batch]
        T_gait: [batch]
        zsw_neg: [batch]
        clipped_step_x: [batch]
        clipped_step_y: [batch]
    Returns:
        p_swing: [batch,3]
    """
    # Vertical Bezier control points (degree 5)
    degree_v = 5
    control_v = torch.stack([
        z_init,          # start height
        z_sw_max / 3,
        z_sw_max,
        z_sw_max,
        z_sw_max / 2,
        zsw_neg,              # negative offset
    ], dim=1)  # [batch,6]

    # Horizontal X and Y (linear interpolation)
    p_swing_x = ((1 - bht) * -clipped_step_x + bht * clipped_step_x).unsqueeze(1)
    p_swing_y = ((1 - bht) * clipped_step_y + bht * clipped_step_y).unsqueeze(1)

    # Z via 5th-degree Bezier
    p_swing_z = bezier_deg(
        0, tau, T_gait, control_v, degree_v
    ).unsqueeze(1)

    v_swing_z = bezier_deg(
        1, tau, T_gait, control_v, degree_v
    ).unsqueeze(1)

    return torch.cat([p_swing_x, p_swing_y, p_swing_z], dim=1), v_swing_z  # [batch,3]


def coth(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.tanh(x)

def precompute_hlip_dynamics(T: float, T_ds: float, z0: float, device: torch.device):
    lam = math.sqrt(9.81 / z0)
    Ts = T - T_ds
    lamTs = lam * Ts
    lamTs_tensor = torch.tensor(lamTs, dtype=torch.float32, device=device)
    sigma1 = lam * coth(0.5 * lamTs_tensor)
    sigma2 = lam * torch.tanh(0.5 * lamTs_tensor)

    cosh_lTs = math.cosh(lamTs)
    sinh_lTs = math.sinh(lamTs)

    A = torch.tensor([[cosh_lTs, sinh_lTs/lam], [lam*sinh_lTs, cosh_lTs]], dtype=torch.float32, device=device)
    B = torch.tensor([[1.0-cosh_lTs], [-lam*sinh_lTs]], dtype=torch.float32, device=device)
    return A, B, sigma1, sigma2


def compute_hlip_orbit_from_dynamics(
    cmd_vel: torch.Tensor,
    T: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    y_nom: float,
    stance_idx: torch.Tensor
):
    """
    Compute batched HLIP desired CoM orbit and foot placement selecting foot based on stance_idx.

    Args:
        cmd_vel: Tensor (N,2) of desired [v_x, v_y].
        T: Tensor (N,) of gait periods.
        A: Tensor (2,2) precomputed dynamics matrix.
        B: Tensor (2,1) precomputed input matrix.
        y_nom: float, nominal lateral foot offset.
        stance_idx: Tensor (N,) with 0 for left stance, 1 for right stance.

    Returns:
        Dict with:
          com_pos_des: Tensor (N,2)
          com_vel_des: Tensor (N,2)
          foot_placement: Tensor (N,2)
    """
    N = cmd_vel.shape[0]
    device = cmd_vel.device

    def Bu(u: torch.Tensor):
        return B.unsqueeze(0) * u.view(N, 1, 1)

    # Forward orbit
    u_x = cmd_vel[:, 0] * T
    X_des = torch.linalg.solve(
        torch.eye(2, device=device).expand(N, 2, 2) - A.unsqueeze(0),
        Bu(u_x)
    )  # (N,2,1)

    # Lateral two-step orbit
    u_left = cmd_vel[:, 1] * T - y_nom
    u_right = cmd_vel[:, 1] * T + y_nom
    A2 = A @ A

    Y_left = torch.linalg.solve(
        torch.eye(2, device=device).expand(N, 2, 2) - A2.unsqueeze(0),
        A.unsqueeze(0) @ Bu(u_left) + Bu(u_right)
    )  # (N,2,1)
    Y_right = torch.linalg.solve(
        torch.eye(2, device=device).expand(N, 2, 2) - A2.unsqueeze(0),
        A.unsqueeze(0) @ Bu(u_right) + Bu(u_left)
    )  # (N,2,1)

    # Select foot target based on stance index

    if stance_idx == 0:
        Y_des = Y_left
        u_y = u_left
    else:
        Y_des = Y_right
        u_y = u_right

    return {
        "com_x": X_des,
        "com_y": Y_des,
        "foot_placement": torch.stack([u_x, u_y], dim=-1),
    }


def compute_desire_com_trajectory(
    cur_time: torch.Tensor,
    Xdesire: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Compute desired COM trajectory relative to stance foot using closed-form HLIP solution.

    Args:
        cur_time: Tensor or float with current time within step.
        Xdesire: Tensor (...,2) initial [pos, vel].
        lam: Natural frequency sqrt(g/z0).

    Returns:
        Tensor (...,2) of desired [pos, vel] at cur_time.
    """
    x0 = Xdesire[..., 0]
    v0 = Xdesire[..., 1]

    pos = x0 * torch.cosh(lam * cur_time) + (v0 / lam) * torch.sinh(lam * cur_time)
    vel = x0 * lam * torch.sinh(lam * cur_time) + v0 * torch.cosh(lam * cur_time)
    return pos, vel
