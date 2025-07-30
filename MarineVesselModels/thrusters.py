import numpy as np


class NaiveDoubleThruster():
    def __init__(
            self,
            b,
    ):
        self.b = b
    
    def newton_to_tau(
            self,
            l_thrust_N,
            r_thrust_N,
    ):
        X = l_thrust_N + r_thrust_N
        N = l_thrust_N*self.b/2 - r_thrust_N*self.b/2
        return np.array([X, 0, N]).reshape([3, 1])