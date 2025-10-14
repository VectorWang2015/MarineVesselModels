import numpy as np
from scipy.linalg import solve_continuous_are


class HeadingLQR():
    def __init__(
            self,
            m33: float,
            d33: float,
            b: float,
            Q: np.ndarray,
            R: np.ndarray,
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

    def step(
            self,
            error_psi: float,
            r: float,
    ) -> float:
        state = np.array([[error_psi], [r]])
        return (self.K@state)[0][0]


class VeloLQR():
    def __init__(
            self,
            m11: float,
            d11: float,
            Q_err: float,
            R_F: float,
            eps_ref: float = 0.,
    ):
        """
        state should be [error_u, tgt_u]
        Q for tgt_u should be 0
        """
        self.A = np.array([[-d11/m11, d11/m11], [0, 0]])
        self.B = np.array([[-2/m11], [0]])
        self.Q = np.diag([Q_err, eps_ref])
        self.R = np.array([[R_F]])

        self.P = solve_continuous_are(a=self.A, b=self.B, q=self.Q, r=self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ self.P

    def step(
            self,
            u: float,
            tgt_u: float,
    ) -> float:
        error_u = tgt_u - u
        state = np.array([[error_u], [tgt_u]])
        return -(self.K@state)[0][0]