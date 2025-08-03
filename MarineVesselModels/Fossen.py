import numpy as np
from typing import Tuple, Optional


class Fossen():
    def __init__(
            self,
            d11,
            d22,
            d33,
            m: Optional[float]=None,
            X_dotu: Optional[float]=None,
            Y_dotv: Optional[float]=None,
            N_dotr: Optional[float]=None,
            I: Optional[float]=None,
            m11: Optional[float]=None,
            m22: Optional[float]=None,
            m33: Optional[float]=None,
    ):
        r"""
        M \dot{v} + Cv + Dv = \tau
            ==> \dot{v} = M^{-1} (\tau - Cv - Dv)
        \dot{x} = v
        assume state is the column vector of [x,y,psi,u,v,r]
        assume tau is the column vector of [X, Y, N]
        """
        self.d11 = d11
        self.d22 = d22
        self.d33 = d33
        self.m = m 

        if m11 is not None and m22 is not None and m33 is not None:
            self.m11 = m11
            self.m22 = m22
            self.m33 = m33
        else:
            self.m11 = m - X_dotu
            self.m22 = m - Y_dotv
            self.m33 = I - N_dotr

        self.M = np.diag([self.m11, self.m22, self.m33])
        self.D = np.diag([d11, d22, d33])

    @property
    def state(self):
        return np.vstack((self.x, self.v))

    def C(self, state):
        u = state[3][0]
        v = state[4][0]
        C = np.array([
            [0, 0, -self.m22*v],
            [0, 0, self.m11*u],
            [self.m22*v, -self.m11*u, 0],
        ])
        return C

    def partial_state(
            self,
            state: np.array,
            tau: np.array,
    ):
        #x = state[:3,:]
        v = state[3:, :]
        u1, u2, r, psi = v[0][0], v[1][0], v[2][0], state[2][0]

        dot_v = np.linalg.inv(self.M) @ (tau - self.C(state)@v - self.D@v)

        # ground coords is related to heading and speed
        dot_x = np.array([
            np.cos(psi)*u1 - np.sin(psi)*u2,
            np.sin(psi)*u1 + np.cos(psi)*u2,
            r,
        ]).reshape([3, 1])
        return np.vstack((dot_x, dot_v))