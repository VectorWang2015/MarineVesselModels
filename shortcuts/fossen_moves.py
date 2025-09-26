import numpy as np
from typing import Optional
from MarineVesselModels.noises import GaussMarkovNoiseGenerator


def noised_fossen_zigzag(
        simulator,
        thruster,
        current_state,
        zigzag_degrees: float,
        base_N: float,
        base_delta_N: float,
) -> np.ndarray:
    init_psi = 0
    current_state = current_state.copy()
    zigzag_psi = zigzag_degrees/180*np.pi

    # turn right
    tgt_psi = init_psi + zigzag_psi
    left_thr = base_N + base_delta_N
    right_thr = base_N - base_delta_N
    tau = thruster.newton_to_tau(left_thr, right_thr)
    states = None
    
    while current_state[2][0] < tgt_psi:
        current_state_with_tau = np.vstack((current_state, tau))
        if states is not None:
            states = np.hstack((states, current_state_with_tau))
        else:
            states = current_state_with_tau.copy()
    
        current_state = simulator.step(tau)
    
    # turn left
    tgt_psi = init_psi - zigzag_psi
    left_thr = base_N - base_delta_N
    right_thr = base_N + base_delta_N
    tau = thruster.newton_to_tau(left_thr, right_thr)
    
    while current_state[2][0] > tgt_psi:
        current_state_with_tau = np.vstack((current_state, tau))
        states = np.hstack((states, current_state_with_tau))
    
        current_state = simulator.step(tau)
    
    # turn right
    tgt_psi = init_psi + zigzag_psi
    left_thr = base_N + base_delta_N
    right_thr = base_N - base_delta_N
    tau = thruster.newton_to_tau(left_thr, right_thr)
    
    while current_state[2][0] < tgt_psi:
        current_state_with_tau = np.vstack((current_state, tau))
        states = np.hstack((states, current_state_with_tau))
    
        current_state = simulator.step(tau)
    
    current_state_with_tau = np.vstack((current_state, tau))
    states = np.hstack((states, current_state_with_tau))

    return states


def fossen_prbs(
        simulator,
        thruster,
        current_state,
        F_l_seq,
        F_r_seq,
) -> np.ndarray:
    states = None
    for F_l, F_r in zip(F_l_seq, F_r_seq):
        tau = thruster.newton_to_tau(F_l, F_r)
        current_state_with_tau = np.vstack((current_state, tau))
        states = current_state_with_tau if states is None else np.hstack((states, current_state_with_tau))
        
        current_state = simulator.step(tau)

    current_state_with_tau = np.vstack((current_state, tau))
    states = current_state_with_tau if states is None else np.hstack((states, current_state_with_tau))

    return states