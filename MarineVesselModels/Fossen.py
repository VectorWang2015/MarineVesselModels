import numpy as np
from typing import Tuple


class LinearFossen():
    def __init__(
            self,
            d11,
            d22,
            d33,
            m ,
            X_dotu,
            Y_dotv,
            N_dotr,
            I,
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
        self.X_dotu = X_dotu
        self.Y_dotv = Y_dotv
        self.N_dotr = N_dotr
        self.I = I

        self.M = np.diag([m-X_dotu, m-Y_dotv, I-N_dotr])
        self.D = np.diag([d11, d22, d33])

    @property
    def state(self):
        return np.vstack((self.x, self.v))

    def C(self, state):
        u = state[3][0]
        v = state[4][0]
        C = np.array([
            [0, 0, -(self.m-self.Y_dotv)*v],
            [0, 0, (self.m-self.X_dotu)*u],
            [(self.m-self.Y_dotv)*v, -(self.m-self.X_dotu)*u, 0],
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
        # ignore C for now
        #dot_v = np.linalg.inv(self.M) @ (tau - self.D@v)

        # ground coords is related to heading and speed
        dot_x = np.array([
            np.cos(psi)*u1 - np.sin(psi)*u2,
            np.sin(psi)*u1 + np.cos(psi)*u2,
            r,
        ]).reshape([3, 1])
        return np.vstack((dot_x, dot_v))