from typing import Dict
import torch
import numpy as np
from scipy.linalg import solve_continuous_are

class CLF:
    """
    Continuous-time Control Lyapunov Function (CLF) evaluator for relative-degree-2 outputs.
    Uses user-provided LIP dynamics (A_lip, B_lip) and augments with double-integrator channels
    for additional outputs. Solves the continuous-time ARE once via SciPy, caches P and LQR gain K
    on the specified torch device for efficient V, V_dot, and control-law evaluation.
    """
    def __init__(
        self,
        n_outputs: int,
        sim_dt: float,
        batch_size: int,
        ordered_output_names: list[str] | None = None,
        device: torch.device = None,
        Q_weights: Dict | np.ndarray | None = None,
        R_weights: Dict | np.ndarray | None = None,
        num_domain: int = 1,
        domain_scalar: list[float]|None = None
    ):
        """Initialize the CLF.

        Args:
            n_outputs: Number of output dimensions.
            sim_dt: Simulation time step (s).
            batch_size: Number of parallel environments.
            ordered_output_names: Ordered list of output names used as keys into
                ``Q_weights`` and ``R_weights`` when those are provided as dicts.
                When ``None`` (or when ``Q_weights``/``R_weights`` are numpy
                arrays), the weight arrays are used directly as the diagonal.
            device: Torch device.
            Q_weights: State cost weights – either a ``dict`` mapping output name
                to ``[q_pos, q_vel]``, or a flat ``np.ndarray`` of length
                ``2 * n_outputs`` (used directly as diagonal).
            R_weights: Input cost weights – either a ``dict`` mapping output name
                to ``[r]``, or a flat ``np.ndarray`` of length ``n_outputs``
                (used directly as diagonal).
            num_domain: Number of CLF domains.
            domain_scalar: Per-domain Q/R scaling factors.
        """
        # Initialize device and basic parameters
        self.device = device
        self.sim_dt = sim_dt
        self.n_outputs = n_outputs
        self.ordered_output_names = ordered_output_names

        # Set up default Q, R if not provided
        self.num_domain = num_domain

        n_states = 2 * n_outputs
        n_inputs = n_outputs

        # Build Q diagonal -------------------------------------------------------
        if Q_weights is None:
            Q = np.ones(n_states)
        elif isinstance(Q_weights, np.ndarray):
            # Legacy: flat array of length n_states passed directly
            Q = Q_weights.astype(float)
        else:
            # Dict branch: requires ordered_output_names
            Q = np.ones(n_states)
            for i, name in enumerate(ordered_output_names):
                Q[2*i] = Q_weights[name][0]
                Q[2*i + 1] = Q_weights[name][1]

        # Build R diagonal -------------------------------------------------------
        if R_weights is None:
            R = 0.1 * np.ones(n_inputs)
        elif isinstance(R_weights, np.ndarray):
            R = R_weights.astype(float)
        else:
            R = np.ones(n_inputs)
            for i, name in enumerate(ordered_output_names):
                R[i] = R_weights[name][0]


        if self.num_domain == 1: 
            
            Q_np = np.diag(Q)
            R_np = np.diag(R)

            # Solve for P and LQR gain K in NumPy
            P_np = self._compute_PK_np(Q_np,R_np)

            # Cache as torch tensors
            self.P = torch.from_numpy(P_np).to(self.device).to(torch.float32)
            self.lambda_max = torch.linalg.eigvalsh(self.P)[-1]
            self.norm_P = torch.linalg.norm(self.P, ord=2)


        else:
            P = []
            lambda_max = []
            norm_P = []
            for i in range(self.num_domain):
                Q_np = np.diag(Q)*domain_scalar[i*2]
                R_np = np.diag(R)*domain_scalar[i*2+1]

                # Solve for P and LQR gain K in NumPy
                P_np = self._compute_PK_np(Q_np,R_np)

                # Cache as torch tensors
                P.append(torch.from_numpy(P_np).to(self.device).to(torch.float32))
                lambda_max.append(torch.linalg.eigvalsh(self.P)[-1])
                norm_P.append(torch.linalg.norm(self.P, ord=2))
            self.P = P

        
        self.v_buffer = torch.zeros((batch_size, 3), device=self.device)
        self.step_count = 0
        
    def _compute_PK_np(self, Q_np, R_np) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct a pure double integrator system for all outputs,
        and solve for the LQR gain K and Lyapunov matrix P.
        """

        # Assume each output has a double integrator model:
        #   [ẋ] = [0 1][x] + [0] u
        #        [0 0]     [1]

        n_outputs = self.n_outputs  # total number of output dimensions (e.g., com x/y/z, foot x/y/z, etc.)

        # 1) Build block-diagonal A and B matrices (double integrators)
        A_blk = np.array([[0.0, 1.0], [0.0, 0.0]])  # (2x2)
        B_blk = np.array([[0.0], [1.0]])           # (2x1)

        A_full = np.kron(np.eye(n_outputs), A_blk)   # (2n x 2n)
        B_full = np.kron(np.eye(n_outputs), B_blk)   # (2n x n)


        # 2) Solve CARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
        P = solve_continuous_are(A_full, B_full, Q_np, R_np)

        return P


    def compute_v(
        self,
        y_act: torch.Tensor,
        y_nom: torch.Tensor,
        dy_act: torch.Tensor,
        dy_nom: torch.Tensor,
        yaw_idx: list[int],
        domain_idx: int = 0,
    ) -> torch.Tensor:
        """
        Evaluate V = (y_act - y_nom)^T P (y_act - y_nom).
        """
        y_err = y_act - y_nom
        dy_err = dy_act - dy_nom
        batch_size = y_act.shape[0]
        eta = torch.zeros(batch_size,2*self.n_outputs, device=y_act.device)
        eta[:,0::2] = y_err      # even indices: positions
        eta[:,1::2] = dy_err     # odd indices: velocities

        #need to wrap around yaw error, 
        yaw_err = y_err[:,yaw_idx]
        two_pi = 2.0 * torch.pi
        wrapped_yaw_err = (yaw_err + torch.pi) % two_pi - torch.pi
        eta[:,yaw_idx] = wrapped_yaw_err

        if self.num_domain > 1:
            P = self.P[domain_idx]
        else:
            P = self.P
        V = torch.einsum('bi,ij,bj->b', eta, P, eta)

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
        yaw_idx: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute V_dot = (V_curr - V_prev) / sim_dt, returns (vdot, V_curr).
        """
        v_curr = self.compute_v(y_act, y_nom, dy_act, dy_nom, yaw_idx)
       
        dt = self.sim_dt
        B = v_curr.shape[0]

        if self.step_count >= 3:
            # We have [V_k, V_{k−1}, V_{k−2}] → 3‐point backward difference
            V_k  = self.v_buffer[:, 0]   # V_k
            V_k1 = self.v_buffer[:, 1]   # V_{k−1}
            V_k2 = self.v_buffer[:, 2]   # V_{k−2}
            # Formula: (3 V_k − 4 V_{k−1} + V_{k−2}) / (2 Δt)
            vdot_raw = (3.0 * V_k - 4.0 * V_k1 + V_k2) / (2.0 * dt)

        elif self.step_count == 2:
            # We only have [V_k, V_{k−1}] → 2‐point fallback
            V_k  = self.v_buffer[:, 0]
            V_k1 = self.v_buffer[:, 1]
            vdot_raw = (V_k - V_k1) / dt

        else:
            # step_count == 1 → no previous sample; just return zero
            vdot_raw = torch.zeros((B,), device=self.device)


        return vdot_raw, v_curr