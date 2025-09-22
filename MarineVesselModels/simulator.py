import numpy as np
from .Fossen import Fossen
from .stepper import time_invariant_simulator_runge_kutta


class VesselSimulator():
    def __init__(
            self,
            hydro_params,
            time_step: float,
            model = Fossen,
            init_state: np.array = np.array([0,0,0,0,0,0]).reshape([6,1]),
            step_fn = time_invariant_simulator_runge_kutta,
            debug = False,
    ):
        self.state = init_state
        self.t = time_step

        self.step_fn = step_fn
        self.model = model(
            **hydro_params,
        )

        self.debug = debug

    def step(
            self,
            tau,
    ):
        new_state = self.step_fn(
            fn_p_state=self.model.partial_state,
            state=self.state,
            tau=tau,
            h=self.t,
        )
        partial = self.model.partial_state(self.state, tau)
        self.state = new_state

        # if debug mode, outputs partial of the state as well
        if self.debug:
            return self.state, partial
        return self.state
