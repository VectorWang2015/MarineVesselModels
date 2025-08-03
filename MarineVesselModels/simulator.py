import numpy as np
from .Fossen import Fossen
from .stepper import time_invariant_simulator_runge_kutta


class FossenSimulator():
    def __init__(
            self,
            hydro_params,
            time_step: float,
            init_state: np.array = np.array([0,0,0,0,0,0]).reshape([6,1]),
            step_fn = time_invariant_simulator_runge_kutta,
    ):
        self.state = init_state
        self.t = time_step

        self.step_fn = step_fn
        self.model = Fossen(
            **hydro_params,
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