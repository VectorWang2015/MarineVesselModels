import numpy as np
from typing import Tuple, Optional
from .noises import gauss_markov_noise


# data from <Research on Parameter Identification Method of Four-Thrusters AUSV Dynamics Model>
# 此参数会自激旋转,可能原因为非线性项缺失
sample_thrust = 5.0
sample_b = 0.5
sample_hydro_params = {
    "d11": 6.0,
    "d22": 7.1,
    "d33": 0.8,
    "m11": 13.0,
    "m22": 23.3,
    "m33": 1.3,
}

# data from <Modeling and Experimental Testing of an UnmannedSurface Vehicle with Rudderless Double Thrusters>
sample_thrust_2 = 100.0
sample_b_2 = 0.52
sample_hydro_params_2 = {
    "m11": 50.05,
    "m22": 84.36,
    "m33": 17.21,
    "X_u": 151.57,
    "Y_v": 132.5,
    "N_r": 34.56,
}
# !!!data suspicous
sample_thrust_params_2 = {
    "c": -1.60e-4,
    "d": 5.04e-3,
}


class Fossen():
    def __init__(
            self,
            m: Optional[float]=None,
            X_dotu: Optional[float]=None,
            Y_dotv: Optional[float]=None,
            N_dotr: Optional[float]=None,
            I: Optional[float]=None,
            m11: Optional[float]=None,
            m22: Optional[float]=None,
            m33: Optional[float]=None,
            d11: Optional[float]=None,
            d22: Optional[float]=None,
            d33: Optional[float]=None,
            X_u: Optional[float]=None,
            Y_v: Optional[float]=None,
            N_r: Optional[float]=None,
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
        if d11 is not None:
            self.D = np.diag([d11, d22, d33])
        else:
            self.D = np.diag([X_u, Y_v, N_r])

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


class FossenWithCurrent(Fossen):
    """
    Linear Fossen that considers current influence
    utilizing current velocity in damping term

    ATTENTION: the influence of waves and winds is supposed to reflect in taus
    """
    def relative_velo(
            self,
            u: float,
            v: float,
            psi: float,
            current_dir: float,
            current_velo: float,
    ) -> Tuple[float, float]:
        """
        calculates relative speed for own vessel concerning current
        
        :param current_dir: direction of the incoming current [rad]
        :param current_velo: ||velocity|| of the current [m/s]
        :param u:
        :param v:
        :param psi:

        :return relative_velocity:
            u_r, v_r, each represents relative velocity components in {b}
        """
        relative_dir = current_dir - psi
        # current u, v in {b}
        u_c = current_velo * np.cos(relative_dir)
        v_c = current_velo * np.sin(relative_dir)

        return u-u_c, v-v_c

    def partial_state(
            self,
            state: np.array,
            tau: np.array,
            current_dir: float=0,
            current_velo: float=0,

    ):
        """
        :param current_dir: direction of the incoming current in {n} [rad]
        :param current_velo: ||velocity|| of the current in {n} [m/s]
        """
        #x = state[:3,:]
        v = state[3:, :]
        u1, u2, r, psi = v[0][0], v[1][0], v[2][0], state[2][0]

        # calculate relative velo in {b} considering current
        u_r, v_r = self.relative_velo(u=u1, v=u2, psi=psi,
                    current_dir=current_dir, current_velo=current_velo)
        relative_v = np.array([u_r, v_r, r]).reshape([3, 1])

        dot_v = np.linalg.inv(self.M) @ (tau - self.C(state)@v - self.D@relative_v)

        # ground coords is related to heading and speed
        dot_x = np.array([
            np.cos(psi)*u1 - np.sin(psi)*u2,
            np.sin(psi)*u1 + np.cos(psi)*u2,
            r,
        ]).reshape([3, 1])
        return np.vstack((dot_x, dot_v))