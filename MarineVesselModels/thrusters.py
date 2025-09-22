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


class NpsThruster():
    def __init__(
            self,
            c,
            d,
    ):
        self.c = c
        self.d = d

    def n_to_newton(
            self,
            n,
            v,
    ):
        """
        n rps
        v sqrt(u**2 + v**2)
        """
        return self.c*v*n + self.d*abs(n)*n


class NpsDoubleThruster(NaiveDoubleThruster):
    def __init__(
            self,
            b,
            c,
            d,
    ):
        self.b = b
        self.l_thrust = NpsThruster(c=c, d=d)
        self.r_thrust = NpsThruster(c=c, d=d)

    def n_to_tau(
            self,
            l_nps,
            r_nps,
            u=0,
            #v=0,
    ):
        # v is small, V = u is a better approximation
        #v = np.sqrt(u**2 + v**2)
        v = u
        l_newton = self.l_thrust.n_to_newton(n=l_nps, v=v)
        r_newton = self.r_thrust.n_to_newton(n=r_nps, v=v)

        return super().newton_to_tau(l_newton, r_newton)
