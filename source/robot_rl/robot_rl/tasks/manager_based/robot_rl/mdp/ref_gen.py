import math
import torch
from torch import Tensor
from typing import Tuple

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
        z_init,                      # Start
        z_init + 0.2 * (z_sw_max - z_init),
        z_init + 0.6 * (z_sw_max - z_init),
        z_sw_max,                    # Peak at mid-swing
        z_init + 0.5 * (z_sw_max - z_init),
        zsw_neg                      # End
    ], dim=1)

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




class HLIP(torch.nn.Module):
    """Hybrid Linear Inverted Pendulum implementation using PyTorch."""

    def __init__(self, grav: float, z0: float, T_ds: float, T: float, y_nom: float):
        super().__init__()
        # Store physical constants
        self.grav = grav
        self.z0 = z0
        self.y_nom = y_nom
        self.lambda_ = torch.sqrt(torch.tensor(grav / z0))
        
        # Get device from grav tensor
        device = torch.tensor(grav).device
        
        # Initialize and store constant matrices on the correct device
        self.A_ss = torch.tensor([[0.0, 1.0], [grav / z0, 0.0]], device=device)
        self.A_ds = torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=device)
        self.B_usw = torch.tensor([-1.0, 0.0], device=device)

        self.T_ds = T_ds
        self.T = T
        self._compute_s2s_matrices()

    def _compute_s2s_matrices(self) -> None:
        """Compute and store step-to-step A and B matrices."""
        # Matrices are already on the correct device from __init__
        exp_ss = torch.matrix_exp(self.A_ss * (self.T - self.T_ds))
        exp_ds = torch.matrix_exp(self.A_ds * self.T_ds)
        self.A_s2s = exp_ss @ exp_ds
        self.B_s2s = exp_ss @ self.B_usw

    def _remap_for_init_stance_state(
        self, X_des_p1: Tensor, Y_des_p2: Tensor, Ux: float, Uy: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Remap the desired state for the initial stance state."""
        # Create tensors on the correct device
        Y_left = torch.cat([
            (Y_des_p2[:, 1, 0] - Uy[:, 1]).unsqueeze(-1),
            Y_des_p2[:, 1, 1].unsqueeze(-1)
        ], dim=-1)  # [batch, 2]
        
        Y_right = torch.cat([
            (Y_des_p2[:, 0, 0] - Uy[:, 0]).unsqueeze(-1),
            Y_des_p2[:, 0, 1].unsqueeze(-1)
        ], dim=-1)  # [batch, 2]
        
        X0 = torch.cat([
            (X_des_p1[:, 0] - Ux).unsqueeze(-1),
            X_des_p1[:, 1].unsqueeze(-1)
        ], dim=-1)  # [batch, 2]
        
        return X0, torch.cat([Y_left.unsqueeze(1), Y_right.unsqueeze(1)], dim=1)  # [batch, 2, 2]

    def _compute_desire_com_trajectory(
        self, cur_time: float, Xdesire: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Compute desired COM trajectory relative to stance foot.
        Args:
            cur_time: float, current time
            Xdesire: Tensor of shape [batch, 2] containing initial position and velocity
        Returns:
            Tuple of (position, velocity) tensors, each of shape [batch]
        """
        x0, v0 = Xdesire[:, 0], Xdesire[:, 1]  # [batch]
        lam = self.lambda_
        pos = x0 * torch.cosh(lam * cur_time) + (v0 / lam) * torch.sinh(lam * cur_time)  # [batch]
        vel = x0 * lam * torch.sinh(lam * cur_time) + v0 * torch.cosh(lam * cur_time)  # [batch]
        return pos, vel

    def _solve_deadbeat_gain(self, A: Tensor, B: Tensor) -> Tensor:
        """Solve for deadbeat gains."""
        A_tmp = torch.stack([
            torch.tensor([-B[0], -B[1]]),
            torch.tensor([
                A[1, 1] * B[0] - A[0, 1] * B[1],
                A[0, 0] * B[1] - A[1, 0] * B[0]
            ])
        ])
        B_tmp = torch.tensor([A[0, 0] + A[1, 1], A[0, 1] * A[1, 0] - A[0, 0] * A[1, 1]])
        return torch.linalg.solve(A_tmp, B_tmp)

    def compute_desired_orbit(
        self,
        vel: Tensor,
        T: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute desired orbit parameters."""
        # Get device from input tensor
        device = vel.device
        
        # Compute matrices if not already done
        if self.A_s2s is None:
            self._compute_s2s_matrices()
            
        # Ensure matrices are on the same device as input
        if self.A_s2s.device != device:
            self.A_s2s = self.A_s2s.to(device)
            self.B_s2s = self.B_s2s.to(device)

        # P1 orbit - handle batch dimension
        U_des_p1 = vel[:, 0] * T  # [batch]
      
        # Expand matrices for batch operations
        eye_expanded = torch.eye(2, device=device).unsqueeze(0).expand(vel.shape[0], -1, -1)  # [batch, 2, 2]
        
        # Solve for each batch element
        X_des_p1 = torch.linalg.solve(eye_expanded- self.A_s2s, self.B_s2s * U_des_p1.unsqueeze(-1))  # [batch, 2, 1]
        X_des_p1 = X_des_p1.squeeze(-1)  # [batch, 2]

        # P2 orbit for left and right
        U_left = vel[:, 1] * T - self.y_nom  # [batch]
        U_right = vel[:, 1] * T + self.y_nom  # [batch]
        
        # Create batch-wise matrices for Y calculations
        A_squared = self.A_s2s @ self.A_s2s  # [batch, 2, 2]
        B_term = self.A_s2s @ self.B_s2s  # [batch, 2]
        
        # Solve for Y_left and Y_right with batch dimension
        Y_left = torch.linalg.solve(
            eye_expanded - A_squared,
            B_term * U_left.unsqueeze(-1) + self.B_s2s * U_right.unsqueeze(-1)
        ).squeeze(-1)  # [batch, 2]
        
        Y_right = torch.linalg.solve(
            eye_expanded - A_squared,
            B_term * U_right.unsqueeze(-1) + self.B_s2s * U_left.unsqueeze(-1)
        ).squeeze(-1)  # [batch, 2]

        return X_des_p1, U_des_p1, torch.stack([Y_left, Y_right], dim=1), torch.stack([U_left, U_right], dim=1)

    def compute_orbit(
        self, T: float, cmd: Tensor
    ) ->  None:
        """Compute desired orbit."""
        # Get desired orbit params
        Xdes, Ux, Ydes, Uy = self.compute_desired_orbit(cmd[:,:2], T)
        # Remap for initial stance
        self.x_init, self.y_init = self._remap_for_init_stance_state(Xdes, Ydes, Ux, Uy)
     
        return Xdes, Ux, Ydes, Uy
   
