import math
import torch
from torch import Tensor
from typing import Tuple
from .ref_gen import coth


class HLIPBatch(torch.nn.Module):
    """Hybrid Linear Inverted Pendulum implementation using PyTorch."""

    def __init__(self, grav: float, z0: float, T_ds: float, T: float, y_nom: float):
        super().__init__()
        # Store physical constants
        self.grav = grav
        self.z0 = z0
        self.y_nom = y_nom
        self.lambda_ = torch.sqrt(torch.tensor(grav / z0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        
        # Get device from grav tensor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize and store constant matrices on the correct device
        self.A_ss = torch.tensor([[0.0, 1.0], [grav / z0, 0.0]], device=device)
        self.A_ds = torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=device)
        self.B_usw = torch.tensor([-1.0, 0.0], device=device)

        self.T_ds = T_ds
        self._compute_s2s_matrices(T)

    def _compute_s2s_matrices(self,T) -> None:
        """Compute and store step-to-step A and B matrices."""
        # Matrices are already on the correct device from __init__
        exp_ss = torch.matrix_exp(self.A_ss.unsqueeze(0) * (T - self.T_ds).unsqueeze(-1).unsqueeze(-1))
        exp_ds = torch.matrix_exp(self.A_ds.unsqueeze(0) * self.T_ds)
        self.A_s2s = exp_ss @ exp_ds
        self.B_s2s = exp_ss @ self.B_usw.unsqueeze(0).unsqueeze(-1)

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

    def compute_desired_orbit(
        self,
        vel: Tensor,
        T: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute desired orbit parameters."""
        # Get device from input tensor
        device = vel.device
        
        # Compute matrices if not already done
        
        self._compute_s2s_matrices(T)
            
        # Ensure matrices are on the same device as input
        if self.A_s2s.device != device:
            self.A_s2s = self.A_s2s.to(device)
            self.B_s2s = self.B_s2s.to(device)

        # P1 orbit - handle batch dimension
        U_des_p1 = vel[:, 0] * T  # [batch]
      
        # Expand matrices for batch operations
        eye_expanded = torch.eye(2, device=device).unsqueeze(0).expand(vel.shape[0], -1, -1)  # [batch, 2, 2]
        
        # Solve for each batch element
        # import pdb; pdb.set_trace()
        rhs = self.B_s2s.squeeze(-1) * U_des_p1.unsqueeze(-1)  # [B, 2]

        X_des_p1 = torch.linalg.solve(eye_expanded - self.A_s2s, rhs).squeeze(-1)

        # P2 orbit for left and right
        U_left = vel[:, 1] * T - self.y_nom  # [batch]
        U_right = vel[:, 1] * T + self.y_nom  # [batch]
        
        # Create batch-wise matrices for Y calculations
        A_squared = self.A_s2s @ self.A_s2s  # [batch, 2, 2]
        B_term = self.A_s2s @ self.B_s2s  # [batch, 2, 1]
        
        rhs_left = B_term.squeeze(-1) * U_left.unsqueeze(-1) + self.B_s2s.squeeze(-1) * U_right.unsqueeze(-1)  # [B, 2]
        rhs_right = B_term.squeeze(-1) * U_right.unsqueeze(-1) + self.B_s2s.squeeze(-1) * U_left.unsqueeze(-1)

        # Solve for Y_left and Y_right with batch dimension
        Y_left = torch.linalg.solve(
            eye_expanded - A_squared,
            rhs_left.unsqueeze(-1)
        ).squeeze(-1)  # [batch, 2]
        
        Y_right = torch.linalg.solve(
            eye_expanded - A_squared,
            rhs_right.unsqueeze(-1)
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
   
