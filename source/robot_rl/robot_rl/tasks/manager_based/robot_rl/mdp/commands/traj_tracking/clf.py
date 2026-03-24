from typing import Dict
import torch
import numpy as np
from scipy.linalg import solve_continuous_are
from isaaclab.utils.math import quat_box_minus

class CLF:
    """
    Continuous-time Control Lyapunov Function (CLF) evaluator for relative-degree-2 outputs.
    Uses user-provided LIP dynamics (A_lip, B_lip) and augments with double-integrator channels
    for additional outputs. Solves the continuous-time ARE once via SciPy, caches P and LQR gain K
    on the specified torch device for efficient V, V_dot, and control-law evaluation.
    """
    def __init__(
        self,
        sim_dt: float,
        batch_size: int,
        ordered_vel_output_names: list[str],
        ordered_pos_output_names: list[str],
        device: torch.device = None,
        Q_weights: Dict = None,
        R_weights: Dict = None,
        # domain_scalar: list[float]|None = None
    ):
        # Initialize device and basic parameters
        self.device = device 
        self.sim_dt = sim_dt
        self.n_outputs = 2*len(ordered_vel_output_names)
        self.ordered_pos_output_names = ordered_pos_output_names
        self.ordered_vel_output_names = ordered_vel_output_names

        # Set up default Q, R if not provided
        # Q_weights should be length = n_states, R_weights length = n_inputs
        # self.num_domain = num_domain

        n_states = self.n_outputs
        self.n_inputs = int(self.n_outputs/2)
        if Q_weights is None:
            Q = np.ones(n_states)
        else:
            # Create the matrix using the dict
            Q = np.ones(n_states)
            for i, name in enumerate(ordered_vel_output_names):
                Q[2*i] = Q_weights[name][0]
                Q[2*i + 1] = Q_weights[name][1]
        if R_weights is None:
            R = 0.1 * np.ones(self.n_inputs)
        else:
            R = np.ones(self.n_inputs)
            for i, name in enumerate(ordered_vel_output_names):
                R[i] = R_weights[name][0]


    # if self.num_domain == 1:

        Q_np = np.diag(Q)
        R_np = np.diag(R)

        # Solve for P and LQR gain K in NumPy
        P_np = self._compute_PK_np(Q_np,R_np)

        # Cache as torch tensors
        self.P = torch.from_numpy(P_np).to(self.device).to(torch.float32)
        self.lambda_max = torch.linalg.eigvalsh(self.P)[-1]
        self.lambda_min = torch.linalg.eigvalsh(self.P)[0]
        self.norm_P = torch.linalg.norm(self.P, ord=2)


        # else:
        #     P = []
        #     lambda_max = []
        #     norm_P = []
        #     for i in range(self.num_domain):
        #         Q_np = np.diag(Q)*domain_scalar[i*2]
        #         R_np = np.diag(R)*domain_scalar[i*2+1]
        #
        #         # Solve for P and LQR gain K in NumPy
        #         P_np = self._compute_PK_np(Q_np,R_np)
        #
        #         # Cache as torch tensors
        #         P.append(torch.from_numpy(P_np).to(self.device).to(torch.float32))
        #         lambda_max.append(torch.linalg.eigvalsh(self.P)[-1])
        #         norm_P.append(torch.linalg.norm(self.P, ord=2))
        #     self.P = P

        # For ablations
        self.P_ID = torch.from_numpy(np.eye(self.n_outputs)).to(self.device).to(torch.float32)

        # Build eta-state indices for each subgroup.
        # For output i, eta has position at 2*i and velocity at 2*i+1.
        subgroup_eta_indices: dict[str, list[int]] = {
            "pelvis_pos": [],
            "pelvis_lin_vel": [],
            "pelvis_ori": [],
            "pelvis_ang_vel": [],
            "joint_pos": [],
            "joint_vel": [],
            "other_body_pos": [],
            "other_body_lin_vel": [],
            "other_body_ori": [],
            "other_body_ang_vel": [],
        }
        for i, name in enumerate(ordered_vel_output_names):
            if name.startswith("pelvis_link:pos_"):
                pos_key = "pelvis_pos"
                vel_key = "pelvis_lin_vel"
            elif name.startswith("pelvis_link:ori_"):
                pos_key = "pelvis_ori"
                vel_key = "pelvis_ang_vel"
            elif name.startswith("joint:"):
                pos_key = "joint_pos"
                vel_key = "joint_vel"
            elif ":pos_" in name:
                pos_key = "other_body_pos"
                vel_key = "other_body_lin_vel"
            elif ":ori_" in name:
                pos_key = "other_body_ori"
                vel_key = "other_body_ang_vel"
            else:
                continue
            subgroup_eta_indices[pos_key].append(2 * i)
            subgroup_eta_indices[vel_key].append(2 * i + 1)

        self.subgroup_indices: dict[str, torch.Tensor] = {
            k: torch.tensor(v, dtype=torch.long, device=self.device)
            for k, v in subgroup_eta_indices.items()
        }
        self.v_subgroups: dict[str, torch.Tensor] = {}

        # Build mapping from body name to [ori_x_idx, ori_y_idx, ori_z_idx, ori_w_idx] in y tensor
        self.ori_body_indices_manifold: dict[str, list[int]] = {}
        self.ori_body_indices_tangent: dict[str, list[int]] = {}

        self.not_quat_mask_manifold = torch.ones(len(ordered_pos_output_names), dtype=torch.bool, device=self.device)
        self.not_quat_mask_tangent = torch.ones(len(ordered_vel_output_names), dtype=torch.bool, device=self.device)

        for i, name in enumerate(ordered_pos_output_names):
            for ori_axis in [":ori_x", ":ori_y", ":ori_z", ":ori_w"]:
                if ori_axis in name:
                    body_name = name.split(ori_axis)[0]

                    if body_name in self.ori_body_indices_manifold:
                        self.ori_body_indices_manifold[body_name].append(i)
                    else:
                        self.ori_body_indices_manifold[body_name] = [i]
                    self.not_quat_mask_manifold[i] = False

        for i, name in enumerate(ordered_vel_output_names):
            for ori_axis in [":ori_x", ":ori_y", ":ori_z"]:
                if ori_axis in name:
                    body_name = name.split(ori_axis)[0]

                    self.not_quat_mask_tangent[i] = False
                    if body_name in self.ori_body_indices_tangent:
                        self.ori_body_indices_tangent[body_name].append(i)
                    else:
                        self.ori_body_indices_tangent[body_name] = [i]

        self.v_buffer = torch.zeros((batch_size, 3), device=self.device)
        self.step_count = 0
        
    def _compute_PK_np(self, Q_np, R_np) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct a pure double integrator system for all outputs,
        and solve for the LQR gain K and Lyapunov matrix P.
        """

        # Assume each output has a double integrator model:
        #   [ẋ] = [0 1][x] + [0] u
        #         [0 0]      [1]


        # 1) Build block-diagonal A and B matrices (double integrators)
        A_blk = np.array([[0.0, 1.0], [0.0, 0.0]])  # (2x2)
        B_blk = np.array([[0.0], [1.0]])           # (2x1)

        A_full = np.kron(np.eye(self.n_inputs), A_blk)   # (2n x 2n)    # n_inputs is totaly number of double integrator systems
        B_full = np.kron(np.eye(self.n_inputs), B_blk)   # (2n x n)


        # 2) Solve CARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
        P = solve_continuous_are(A_full, B_full, Q_np, R_np)

        return P


    def compute_v(
        self,
        y_act: torch.Tensor,
        y_nom: torch.Tensor,
        dy_act: torch.Tensor,
        dy_nom: torch.Tensor,
        domain_idx: int = 0,
    ) -> torch.Tensor:
        """
        Evaluate V = (y_act - y_nom)^T P (y_act - y_nom).
        """

        y_err = self.compute_y_err(y_act, y_nom)

        dy_err = dy_act - dy_nom
        batch_size = y_act.shape[0]
        eta = torch.zeros(batch_size,self.n_outputs, device=y_act.device)
        eta[:,0::2] = y_err      # even indices: positions
        eta[:,1::2] = dy_err     # odd indices: velocities

        # Compute per-subgroup V contributions (self-contribution, excluding cross-terms)
        self.v_subgroups = {}
        for name, idx in self.subgroup_indices.items():
            if idx.numel() > 0:
                eta_sub = eta[:, idx].contiguous()
                P_sub = self.P_ID[idx][:, idx].contiguous()     # Use identity P for these to just compute errors, not Lyap values
                self.v_subgroups[name] = (torch.matmul(eta_sub, P_sub) * eta_sub).sum(dim=-1)

        V = torch.einsum('bi,ij,bj->b', eta, self.P, eta)

        self.v_buffer[:, 2] = self.v_buffer[:, 1]
        self.v_buffer[:, 1] = self.v_buffer[:, 0]
        # We detach() so that backprop does not try to flow through the history buffer.
        self.v_buffer[:, 0] = V.detach()

        self.step_count += 1
        return V

    def compute_vdot(
        self,
        y_act: torch.Tensor,
        y_nom: torch.Tensor,
        dy_act: torch.Tensor,
        dy_nom: torch.Tensor,
        # yaw_idx: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute V_dot = (V_curr - V_prev) / sim_dt, returns (vdot, V_curr).
        """
        v_curr = self.compute_v(y_act, y_nom, dy_act, dy_nom,)# yaw_idx)
       
        dt = self.sim_dt    # TODO: This dt is wrong - should probably be more like sim_dt*decimation
        # print(f"dt={dt}")
        B = v_curr.shape[0]

        # if self.step_count >= 3:
        #     # We have [V_k, V_{k−1}, V_{k−2}] → 3‐point backward difference
        #     V_k  = self.v_buffer[:, 0]   # V_k
        #     V_k1 = self.v_buffer[:, 1]   # V_{k−1}
        #     V_k2 = self.v_buffer[:, 2]   # V_{k−2}
        #     # Formula: (3 V_k − 4 V_{k−1} + V_{k−2}) / (2 Δt)
        #     vdot_raw = (3.0 * V_k - 4.0 * V_k1 + V_k2) / (2.0 * dt)

        # if self.step_count >= 2:
        # We only have [V_k, V_{k−1}] → 2‐point fallback
        V_k  = self.v_buffer[:, 0]
        V_k1 = self.v_buffer[:, 1]
        vdot_raw = (V_k - V_k1) #/ dt

        # else:
        #     # step_count == 1 → no previous sample; just return zero
        #     vdot_raw = torch.zeros((B,), device=self.device)

        # Clamp unreasonably large vdot values (e.g., after resets)
        vdot_raw = torch.where(torch.abs(vdot_raw) > 10000, torch.zeros_like(vdot_raw), vdot_raw)

        return vdot_raw, v_curr


    def compute_vdot_analytic(
        self,
        y_act: torch.Tensor,
        y_nom: torch.Tensor,
        dy_act: torch.Tensor,
        dy_nom: torch.Tensor,
        ddy_act: torch.Tensor,
        ddy_nom: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute V_dot analytically using acceleration information.

        Uses the formula: Vdot = 2 * eta^T * P * eta_dot

        where:
            eta = [y_err_0, dy_err_0, y_err_1, dy_err_1, ...] (interleaved pos/vel errors)
            eta_dot = [dy_err_0, ddy_err_0, dy_err_1, ddy_err_1, ...] (interleaved vel/acc errors)

        Args:
            y_act: [B, n_outputs] actual positions.
            y_nom: [B, n_outputs] nominal/desired positions.
            dy_act: [B, n_outputs] actual velocities.
            dy_nom: [B, n_outputs] nominal/desired velocities.
            ddy_act: [B, n_outputs] actual accelerations (from simulation).
            ddy_nom: [B, n_outputs] nominal/desired accelerations (from trajectory).
            yaw_idx: list of indices for yaw outputs that need angle wrapping.
            domain_idx: domain index for multi-domain P matrices.

        Returns:
            (vdot, v_curr): tuple of Vdot and V values, each shape [B].
        """
        # Compute errors
        y_err = self.compute_y_err(y_act, y_nom)
        dy_err = dy_act - dy_nom
        ddy_err = ddy_act - ddy_nom

        batch_size = y_act.shape[0]

        # Build eta: [y_err_0, dy_err_0, y_err_1, dy_err_1, ...]
        eta = torch.zeros(batch_size, self.n_outputs, device=y_act.device)
        eta[:, 0::2] = y_err      # even indices: position errors
        eta[:, 1::2] = dy_err     # odd indices: velocity errors

        # # Wrap yaw errors to [-pi, pi]
        # yaw_err = y_err[:, yaw_idx]
        # two_pi = 2.0 * torch.pi
        # wrapped_yaw_err = (yaw_err + torch.pi) % two_pi - torch.pi
        # # Update the position error slots for yaw indices in eta
        # yaw_eta_indices = [2 * i for i in yaw_idx]
        # eta[:, yaw_eta_indices] = wrapped_yaw_err

        # Build eta_dot: [dy_err_0, ddy_err_0, dy_err_1, ddy_err_1, ...]
        eta_dot = torch.zeros(batch_size, self.n_outputs, device=y_act.device)
        eta_dot[:, 0::2] = dy_err     # even indices: velocity errors (d/dt of position error)
        eta_dot[:, 1::2] = ddy_err    # odd indices: acceleration errors (d/dt of velocity error)

        # Select P matrix based on domain
        # if self.num_domain > 1:
        #     P = self.P[domain_idx]
        # else:
        P = self.P

        # Compute V = eta^T P eta
        V = torch.einsum('bi,ij,bj->b', eta, P, eta)

        # Compute per-subgroup V contributions (for logging/debugging)
        self.v_subgroups = {}
        for name, idx in self.subgroup_indices.items():
            if idx.numel() > 0:
                eta_sub = eta[:, idx].contiguous()
                P_sub = P[idx][:, idx].contiguous()
                self.v_subgroups[name] = (torch.matmul(eta_sub, P_sub) * eta_sub).sum(dim=-1)

        # Compute Vdot = 2 * eta^T * P * eta_dot
        # First compute P @ eta_dot: [B, 2n]
        P_eta_dot = torch.matmul(eta_dot, P.T)  # [B, 2n]
        # Then compute eta^T @ (P @ eta_dot): scalar per batch
        vdot = 2.0 * torch.einsum('bi,bi->b', eta, P_eta_dot)   # TODO: Check this

        # Update V buffer for consistency with finite differencing method
        self.v_buffer[:, 2] = self.v_buffer[:, 1]
        self.v_buffer[:, 1] = self.v_buffer[:, 0]
        self.v_buffer[:, 0] = V.detach()
        self.step_count += 1

        return vdot, V

    def compute_y_err(self, y_act, y_des):
        """
        Compute the error between the measured values and the desired values.

        Must respect the quaternion orientation by using the box minus operator.
        """
        y_err = torch.zeros((y_act.shape[0], len(self.ordered_vel_output_names)), device=self.device)

        # Subtract all the non-quaternions normally
        y_err[:, self.not_quat_mask_tangent] = y_act[:, self.not_quat_mask_manifold] - y_des[:, self.not_quat_mask_manifold]

        # Subtract quats with box minus operator
        for body_name, indices in self.ori_body_indices_manifold.items():
            tangent_indices = self.ori_body_indices_tangent[body_name]
            y_err[:, tangent_indices] = quat_box_minus(y_act[:, indices], y_des[:, indices])

        return y_err