import numpy as np
from .Fossen import LinearFossen
from .stepper import time_invariant_simulator_runge_kutta


class LinearFossenSimulator():
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
            time_step: float,
            init_state: np.array = np.array([0,0,0,0,0,0]).reshape([6,1]),
            step_fn = time_invariant_simulator_runge_kutta,
    ):
        self.d11 = d11
        self.d22 = d22
        self.d33 = d33
        self.m = m 
        self.X_dotu = X_dotu
        self.Y_dotv = Y_dotv
        self.N_dotr = N_dotr
        self.I = I

        self.state = init_state
        self.t = time_step

        self.step_fn = step_fn
        self.model = LinearFossen(
            d11,
            d22,
            d33,
            m ,
            X_dotu,
            Y_dotv,
            N_dotr,
            I,
        )

    def step(
            self,
            tau,
    ):
        self.state = self.step_fn(
            fn_p_state=self.model.partial_state,
            state=self.state,
            tau=tau,
            h=self.t,
        )

        return self.state