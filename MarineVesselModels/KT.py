import numpy as np
from typing import Tuple, Optional


# data from <基于改进最小二乘算法的船舶操纵性参数辨识>
sample_hydro_params = {
    "n": 10.23,
    "u": 1.179,
    "K": 0.216,
    "T": 13.514,
    "alpha": 258.405,
}


class FirstOrderResponse():
    def __init__(
            self,
            K: float,
            T: float,
            alpha: float,
            u: float,
            **kwargs,
    ):
        r"""
        T \dot{r} + r + \alpha r^3 = K \delta
        """
        self.K = K
        self.T = T
        self.alpha = alpha
        self.u = u

    def partial_state(
            self,
            state: np.array,
            tau: np.array,
    ):
        """
        tau should be rudder angle delta
        """
        #x = state[:3,:]
        v = state[3:, :]
        u1, u2, r, psi = v[0][0], v[1][0], v[2][0], state[2][0]
        delta = tau[0][0]

        assert u1 == self.u
        assert u2 == 0

        dot_u1, dot_u2 = 0, 0
        dot_r = (self.K * delta - self.alpha * np.power(r, 3) - r) / self.T
        dot_v = np.array((dot_u1, dot_u2, dot_r)).reshape((3, 1))

        # ground coords is related to heading and speed
        dot_x = np.array([
            np.cos(psi)*u1 - np.sin(psi)*u2,
            np.sin(psi)*u1 + np.cos(psi)*u2,
            r,
        ]).reshape([3, 1])
        return np.vstack((dot_x, dot_v))


class NoisyFirstOrderResponse(FirstOrderResponse):
    def __init__(
            self,
            K: float,
            T: float,
            alpha: float,
            u: float,
            variance: float=0.05,
            **kwargs,
    ):
        r"""
        T \dot{r} + r + \alpha r^3 = K \delta
        """
        self.K = K
        self.T = T
        self.alpha = alpha
        self.u = u
        self.variance = variance

    def partial_state(
            self,
            state: np.array,
            tau: np.array,
    ):
        dot_state = super().partial_state(state, tau)
        dot_state[5][0] += np.random.normal(0, self.variance)
        return dot_state