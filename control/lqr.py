import numpy as np
from scipy.linalg import solve_continuous_are
from typing import Optional


class HeadingLQR():
    def __init__(
            self,
            m33: float,
            d33: float,
            b: float,
            Q: np.ndarray,
            R: np.ndarray,
            control_lim: Optional[float] = None,
    ):
        """
        state: [tgt_psi - psi, r]
        """
        assert Q.shape[0] == 2
        assert Q.shape[0] == Q.shape[1]
        assert R.shape[0] == 1
        assert R.shape[0] == R.shape[1]

        self.A = np.array([[0, -1], [0, -d33/m33]])
        self.B = np.array([[0], [b/m33]])

        self.P = solve_continuous_are(a=self.A, b=self.B, q=Q, r=R)
        self.K = - np.linalg.inv(R) @ self.B.T @ self.P

        self.control_lim = control_lim

    def step(
            self,
            error_psi: float,
            r: float,
    ) -> float:
        state = np.array([[error_psi], [r]])
        u = (self.K@state)[0][0]
        if self.control_lim is None:
            return u
        else:
            return max(min(u, self.control_lim), - self.control_lim)


class VeloLQR():
    def __init__(
            self,
            m11: float,
            d11: float,
            Q_err: float,
            R_F: float,
            control_lim: Optional[float] = None,
    ):
        """
        state should be [error_u]
        use Forward-feed
        """
        self.d11 = d11
        self.m11 = m11
        self.A = np.array([[-d11/m11]])
        self.B = np.array([[-2/m11]])
        self.Q = np.diag([[Q_err]])
        self.R = np.array([[R_F]])

        self.P = solve_continuous_are(a=self.A, b=self.B, q=self.Q, r=self.R)
        self.K = - np.linalg.inv(self.R) @ self.B.T @ self.P

        self.control_lim = control_lim

    def step(
            self,
            u: float,
            tgt_u: float,
    ) -> float:
        error_u = tgt_u - u
        state = np.array([[error_u]])

        ff = self.d11 / 2 * tgt_u
        control = ff + (self.K@state)[0][0]

        if self.control_lim is None:
            return control
        else:
            return max(min(control, self.control_lim), - self.control_lim)