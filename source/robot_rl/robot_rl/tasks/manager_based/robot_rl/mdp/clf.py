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
        A_lip: torch.Tensor,
        B_lip: torch.Tensor,
        n_outputs: int,
        sim_dt: float,
        device: torch.device = None,
        Q_weights: np.ndarray = None,
        R_weights: np.ndarray = None,
    ):
        # Initialize device and basic parameters
        self.device = device 
        self.sim_dt = sim_dt
        self.n_outputs = n_outputs

        # Convert LIP dynamics to NumPy
        self.A_lip = A_lip.cpu().numpy()
        self.B_lip = B_lip.cpu().numpy()

        # Set up default Q, R if not provided
        # Q_weights should be length = n_states, R_weights length = n_inputs
        n_states = 2 * 2 + 2 * (n_outputs - 2)
        n_inputs = 2 + (n_outputs - 2)
        if Q_weights is None:
            Q_weights = np.ones(n_states)
        if R_weights is None:
            R_weights = 0.1 * np.ones(n_inputs)
        self.Q_np = np.diag(Q_weights)
        self.R_np = np.diag(R_weights)

        # Solve for P and LQR gain K in NumPy
        P_np, K_np = self._compute_PK_np()

        # Cache as torch tensors
        self.P = torch.from_numpy(P_np).to(self.device).to(torch.float32)
        # K shape: (n_inputs, n_states)
        self.K = torch.from_numpy(K_np).to(self.device)

    def _compute_PK_np(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build full state-space (A_full, B_full) in NumPy, solve continuous ARE,
        and compute LQR gain K = R^{-1} B^T P.
        """
        # 1) Build LIP x & y blocks
        A_xy = np.block([
            [self.A_lip, np.zeros_like(self.A_lip)],
            [np.zeros_like(self.A_lip), self.A_lip]
        ])  # (4×4)
        B_xy = np.block([
            [self.B_lip, np.zeros_like(self.B_lip)],
            [np.zeros_like(self.B_lip), self.B_lip]
        ])  # (4×2)

        # 2) Double-integrators for extra outputs
        n_extra = self.n_outputs - 2
        A_blk = np.array([[0.0, 1.0], [0.0, 0.0]])
        B_blk = np.array([[0.0], [1.0]])
        A_extra = np.kron(np.eye(n_extra), A_blk)  # (2*n_extra × 2*n_extra)
        B_extra = np.kron(np.eye(n_extra), B_blk)  # (2*n_extra × n_extra)

        # 3) Assemble full matrices
        A_full = np.block([
            [A_xy,             np.zeros((A_xy.shape[0], A_extra.shape[1]))],
            [np.zeros((A_extra.shape[0], A_xy.shape[1])), A_extra]
        ])
        B_full = np.block([
            [B_xy,             np.zeros((B_xy.shape[0], B_extra.shape[1]))],
            [np.zeros((B_extra.shape[0], B_xy.shape[1])), B_extra]
        ])

        # 4) Solve CARE for P
        P = solve_continuous_are(A_full, B_full, self.Q_np, self.R_np)

        # 5) Compute LQR gain K = R^{-1} B^T P
        K = np.linalg.solve(self.R_np, B_full.T.dot(P))

        return P, K

    def compute_v(
        self,
        y_act: torch.Tensor,
        y_nom: torch.Tensor,
        dy_act: torch.Tensor,
        dy_nom: torch.Tensor,
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

        V = torch.einsum('bi,ij,bj->b', eta, self.P, eta)
        return V

    def compute_vdot(
        self,
        y_act: torch.Tensor,
        y_nom: torch.Tensor,
        dy_act: torch.Tensor,
        dy_nom: torch.Tensor,
        v_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute V_dot = (V_curr - V_prev) / sim_dt, returns (vdot, V_curr).
        """
        v_curr = self.compute_v(y_act, y_nom,dy_act,dy_nom)
        vdot = (v_curr - v_prev) / self.sim_dt
        return vdot, v_curr