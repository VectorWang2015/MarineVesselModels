import numpy as np
from typing import Dict
from scipy.signal import savgol_filter


class WeightedRidgeLSFossen:
    r"""
    Improvements:
      - SG smooth/differential
      - column normalization
      - WLS
      - Ridge regularization
    """

    def __init__(
        self,
        time_step: float,
        SG_window: int = 11,
        weights=(1.0, 1, 2.5),   # (Fx, Fy, Nr) weight
        lam: float = 1e-3,          # ridge regularization weight
        enable_filter: bool = True,
    ):
        self.time_step = time_step
        self.SG_window = SG_window
        self.weights = weights
        self.lam = lam
        self.scale = None
        self.theta = None

        self.enable_filter = enable_filter

    def _theta_to_params(self, theta: np.array) -> Dict[str, float]:
        return {
            "m11": theta[0, 0],
            "m22": theta[1, 0],
            "m33": theta[2, 0],
            "d11": theta[3, 0],
            "d22": theta[4, 0],
            "d33": theta[5, 0],
        }

    def smooth_fn(self, signals):
        return savgol_filter(signals, window_length=self.SG_window, polyorder=3)

    def dot_fn(self, signals):
        return savgol_filter(
            signals, window_length=self.SG_window,
            polyorder=3, deriv=1, delta=self.time_step
        )

    def construct_matrices(
            self,
            us: np.array,
            vs: np.array,
            rs: np.array,
            dot_us: np.array,
            dot_vs: np.array,
            dot_rs: np.array,
            taus: np.array,
    ) -> Dict[str, float]:

        if self.enable_filter:
            us = self.smooth_fn(us)
            vs = self.smooth_fn(vs)
            rs = self.smooth_fn(rs)
    
            dot_us = self.smooth_fn(dot_us)
            dot_vs = self.smooth_fn(dot_vs)
            dot_rs = self.smooth_fn(dot_rs)

        Hs, Ys = [], []
        for i in range(us.shape[0] - 1):
            u, v, r = us[i], vs[i], rs[i]
            dot_u, dot_v, dot_r = dot_us[i], dot_vs[i], dot_rs[i]
            tau = taus[:,i].reshape([3, 1])

            H_i = np.array([
                [dot_u, -v*r, 0, u, 0, 0],
                [u*r, dot_v, 0, 0, v, 0],
                [-u*v, u*v, dot_r, 0, 0, r],
            ])

            Hs.append(H_i)
            Ys.append(tau)

        H = np.vstack(Hs)
        y = np.vstack(Ys)

        self.H = H
        self.y = y

        return H, y

    def identificate(
            self,
            us: np.array,
            vs: np.array,
            rs: np.array,
            dot_us: np.array,
            dot_vs: np.array,
            dot_rs: np.array,
            taus: np.array,
    ) -> Dict[str, float]:
        H, y = self.construct_matrices(us, vs, rs, dot_us, dot_vs, dot_rs, taus)

        # column norm, 1e-12 for perventing 0
        self.scale = np.linalg.norm(H, axis=0) + 1e-12
        Hs = H / self.scale

        # channel weight
        w_x, w_y, w_r = self.weights
        N = us.shape[0] - 1
        W = np.diag([w_x, w_y, w_r]*N)

        # ridge regularization
        A = Hs.T @ W @ Hs + self.lam * np.eye(Hs.shape[1])
        b = Hs.T @ W @ y
        # linalg.solve soves a well-determined full-rank, linear matrix equation Ax=b
        theta_s = np.linalg.solve(A, b)

        # denorm
        self.theta = theta_s / self.scale.reshape(-1, 1)

        return self._theta_to_params(self.theta)
    
    def compute_residuals(self) -> Dict[str, float]:
        """
        returns residuals for each channel: Fx, Fy, Nr
        """
        if self.theta is None or self.H is None or self.y is None:
            raise RuntimeError("call identificate() first!")

        y_pred = self.H @ self.theta
        residuals = self.y - y_pred

        res_fx = residuals[0::3]
        res_fy = residuals[1::3]
        res_nr = residuals[2::3]

        rms_fx = np.sqrt(np.mean(res_fx**2))
        rms_fy = np.sqrt(np.mean(res_fy**2))
        rms_nr = np.sqrt(np.mean(res_nr**2))

        return {"Fx_RMS": rms_fx, "Fy_RMS": rms_fy, "Nr_RMS": rms_nr}

