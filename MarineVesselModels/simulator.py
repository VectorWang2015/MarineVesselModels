import numpy as np
from .Fossen import Fossen
from .stepper import time_invariant_simulator_runge_kutta
from .noises import GaussianNoiseGenerator, GaussMarkovNoiseGenerator

from typing import Optional


class VesselSimulator():
    def __init__(
            self,
            hydro_params,
            time_step: float,
            model = Fossen,
            init_state: np.array = np.array([0,0,0,0,0,0]).reshape([6,1]),
            step_fn = time_invariant_simulator_runge_kutta,
            output_partial = False,
    ):
        self.state = init_state
        self.t = time_step

        self.step_fn = step_fn
        self.model = model(
            **hydro_params,
        )

        self.output_partial = output_partial

        self.states = [init_state]
        self.raw_taus = []
        self.partials = []

    def step(
            self,
            tau,
    ):
        self.raw_taus.append(tau)

        new_state = self.step_fn(
            fn_p_state=self.model.partial_state,
            state=self.state,
            tau=tau,
            h=self.t,
        )
        partial = self.model.partial_state(self.state, tau)
        self.state = new_state

        self.states.append(self.state)
        self.partials.append(partial)

        # if debug mode, outputs partial of the state as well
        if self.output_partial:
            return self.state, partial
        return self.state

    def summarize(
            self,
    ):
        return {
            "states": self.states,
            "raw_taus": self.raw_taus,
            "partials": self.partials,
        }


class NoisyVesselSimulator(VesselSimulator):
    def __init__(
            self,
            hydro_params,
            time_step: float,
            model = Fossen,
            init_state: np.array = np.array([0,0,0,0,0,0]).reshape([6,1]),
            step_fn = time_invariant_simulator_runge_kutta,
            output_partial = False,
            tau_noise_gen: Optional[GaussMarkovNoiseGenerator] = None,
            obsv_noise_gen: Optional[GaussianNoiseGenerator] = None,
    ):
        super().__init__(hydro_params, time_step, model, init_state, step_fn, output_partial)
        self.tau_noise_gen = tau_noise_gen
        self.obsv_noise_gen = obsv_noise_gen

        self.tau_noises = []
        self.obsv_noises = []
        self.obsvs = []

    def step(
            self,
            tau,
    ):
        self.raw_taus.append(tau)

        if self.tau_noise_gen is not None:
            t_noise = self.tau_noise_gen.step().reshape(tau.shape)
        else:
            t_noise = np.zeros(tau.shape)

        self.tau_noises.append(t_noise)

        new_state = self.step_fn(
            fn_p_state=self.model.partial_state,
            state=self.state,
            tau=tau+t_noise,
            h=self.t,
        )
        # parital is model's parital, not RK method's partial that used to update
        partial = self.model.partial_state(self.state, tau)
        self.state = new_state

        if self.obsv_noise_gen is not None:
            s_noise = self.obsv_noise_gen.step().reshape(self.state.shape)
        else:
            s_noise = np.zeros(self.state.shape)

        self.obsv_noises.append(s_noise)
        obsv = self.state + s_noise

        self.obsvs.append(obsv)
        self.states.append(self.state)
        self.partials.append(partial)

        # if debug mode, outputs partial of the state as well
        if self.output_partial:
            return obsv, partial
        return obsv

    def summarize(
            self,
    ):
        return {
            "raw_taus": self.raw_taus,
            "tau_noises": self.tau_noises,
            "states": self.states,
            "partials": self.partials,
            "obsvs": self.obsvs,
            "obsv_noises": self.obsv_noises,
        }