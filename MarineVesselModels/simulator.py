import numpy as np
from .Fossen import Fossen, FossenWithCurrent
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


class SimplifiedEnvironmentalDisturbanceSimulator(VesselSimulator):
    """
    TODO:
        this class should be refactoried to use env force generator and current generator,
        so that it can utilize dynamic environmental disturbance
    """
    def __init__(
            self,
            hydro_params,
            time_step: float,
            model = FossenWithCurrent,
            init_state: np.array = np.array([0,0,0,0,0,0]).reshape([6,1]),
            step_fn = time_invariant_simulator_runge_kutta,
            output_partial = False,
            env_force_magnitude: float=0,
            env_force_direction: float=0,  # radians
            current_velocity: float=0,  # [m/s]
            current_direction: float=0, # [rad]
    ):
        super().__init__(hydro_params, time_step, model, init_state, step_fn, output_partial)
        self.env_force_magnitude = env_force_magnitude
        self.env_force_direction = env_force_direction
        self.current_velocity = current_velocity
        self.current_direction = current_direction
        
        # Pre-compute global force vector (no moment component for now)
        self.env_force_global = np.array([
            env_force_magnitude * np.cos(env_force_direction),
            env_force_magnitude * np.sin(env_force_direction),
            0.0  # No environmental yaw moment in simplified model
        ]).reshape([3, 1])
        
        self.env_forces_body = []  # Store environmental forces in body frame for analysis
        self.applied_taus = []     # Store total tau (control + environmental) applied to model

    def step(
            self,
            tau,
    ):
        self.raw_taus.append(tau)
        
        # Get current heading
        psi = self.state[2][0]
        
        # Convert global environmental force to body frame
        # Rotation matrix from body to global: R(psi) = [[cos(psi), -sin(psi)], [sin(psi), cos(psi)]]
        # So global vector v_global = R(psi) @ v_body => v_body = R(psi)^T @ v_global
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        R_T = np.array([
            [cos_psi, sin_psi, 0],
            [-sin_psi, cos_psi, 0],
            [0, 0, 1]
        ])
        
        env_force_body = R_T @ self.env_force_global
        self.env_forces_body.append(env_force_body)
        
        # Combine control input with environmental disturbance
        total_tau = tau + env_force_body
        self.applied_taus.append(total_tau)
        
        # passing current information to model
        if self.current_velocity != 0:
            current_information = {
                "current_dir": self.current_direction,
                "current_velo": self.current_velocity,
            }
    
            new_state = self.step_fn(
                fn_p_state=self.model.partial_state,
                state=self.state,
                tau=total_tau,
                h=self.t,
                fn_kwargs=current_information,
            )
            partial = self.model.partial_state(self.state, total_tau, **current_information)
        # for history demo compatibility
        else:
            new_state = self.step_fn(
                fn_p_state=self.model.partial_state,
                state=self.state,
                tau=total_tau,
                h=self.t,
            )
            partial = self.model.partial_state(self.state, total_tau)

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
        base_summary = super().summarize()
        # Override raw_taus with applied_taus for clarity
        return {
            "raw_taus": self.raw_taus,  # Original control inputs
            "applied_taus": self.applied_taus,  # Control + environmental forces
            "env_forces_body": self.env_forces_body,  # Environmental forces in body frame
            "states": self.states,
            "partials": self.partials,
            "env_force_magnitude": self.env_force_magnitude,
            "env_force_direction": self.env_force_direction,
            "env_force_global": self.env_force_global,
        }