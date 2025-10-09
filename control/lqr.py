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
        assert Q.shape[0] == 2
        assert Q.shape[0] == Q.shape[1]
        assert R.shape[0] == 1
        assert R.shape[0] == R.shape[1]

        self.A = np.array([[0, 1], [0, -d33/m33]])
        self.B = np.array([[0], [b/m33]])

        self.P = solve_continuous_are(a=self.A, b=self.B, q=Q, r=R)
        self.K = np.linalg.inv(R) @ self.B.T @ self.P

    def control(
            self,
            error_delta: float,
            r: float,
    ) -> float:
        state = np.array([[error_delta], [r]])
        return (self.K@state)[0][0]