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
   

        # Set up default Q, R if not provided
        # Q_weights should be length = n_states, R_weights length = n_inputs
        n_states = 2 * n_outputs
        n_inputs = n_outputs
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
        P = solve_continuous_are(A_full, B_full, self.Q_np, self.R_np)

        # 3) Compute LQR gain: K = R^{-1} B^T P
        K = np.linalg.solve(self.R_np, B_full.T @ P)

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