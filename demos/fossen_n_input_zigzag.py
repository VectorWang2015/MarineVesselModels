import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.thrusters import NpsThruster, NpsDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_params_2, Fossen


def fossen_nps_zigzag(
        simulator,
        thruster,
        current_state,
        zigzag_degrees: float,
        base_nps: float,
        delta_nps: float,
):
    init_psi = 0
    current_state = current_state.copy()
    zigzag_psi = zigzag_degrees/180*np.pi

    # turn right
    tgt_psi = init_psi + zigzag_psi
    left_thr = base_nps + delta_nps
    right_thr = base_nps - delta_nps
    states = None
    
    while current_state[2][0] < tgt_psi:
        u = current_state[3][0]
        v = current_state[4][0]
        tau = thruster.n_to_tau(left_thr, right_thr, u)

        nps = np.array([left_thr, right_thr]).reshape([2, 1])
        current_state_with_nps = np.vstack((current_state, nps))
        if states is not None:
            states = np.hstack((states, current_state_with_nps))
        else:
            states = current_state_with_nps.copy()
    
        current_state = simulator.step(tau)
    
    # turn left
    tgt_psi = init_psi - zigzag_psi
    left_thr, right_thr = right_thr, left_thr
    
    while current_state[2][0] > tgt_psi:
        u = current_state[3][0]
        v = current_state[4][0]
        tau = thruster.n_to_tau(left_thr, right_thr, u)

        nps = np.array([left_thr, right_thr]).reshape([2, 1])
        current_state_with_nps = np.vstack((current_state, nps))
        states = np.hstack((states, current_state_with_nps))
    
        current_state = simulator.step(tau)
    
    # turn right
    tgt_psi = init_psi + zigzag_psi
    left_thr, right_thr = right_thr, left_thr
    
    while current_state[2][0] < tgt_psi:
        u = current_state[3][0]
        v = current_state[4][0]
        tau = thruster.n_to_tau(left_thr, right_thr, u)

        nps = np.array([left_thr, right_thr]).reshape([2, 1])
        current_state_with_nps = np.vstack((current_state, nps))
        states = np.hstack((states, current_state_with_nps))
    
        current_state = simulator.step(tau)
    
    nps = np.array([left_thr, right_thr]).reshape([2, 1])
    current_state_with_nps = np.vstack((current_state, nps))
    states = np.hstack((states, current_state_with_nps))

    return states


if __name__ == "__main__":
    # test thrust
    # 100rps => approx 50N, 120rps => approx 72N
    # test_thrust = NpsThruster(**sample_thrust_params_2)

    # npss = list(np.arange(-200, 210, 10))
    # newtons = [test_thrust.n_to_newton(n, -20) for n in npss]

    # zigzag 10
    # converged u under base thrust 50N: 0.659761
    time_step = 0.01
    
    init_psi = 0
    base_nps = 100
    delta_nps = 20.0
    zigzag_degrees = 10.0
    
    current_state = np.array([0, 0, init_psi, 0.659761, 0, 0]).reshape([6, 1])
    
    simulator = VesselSimulator(
        sample_hydro_params_2,
        time_step=time_step,
        model=Fossen,
        init_state=current_state,
    )
    thruster = NpsDoubleThruster(b=sample_b_2, **sample_thrust_params_2)
    
    states = fossen_nps_zigzag(
        simulator=simulator,
        thruster=thruster,
        current_state=current_state,
        zigzag_degrees=zigzag_degrees,
        base_nps=base_nps,
        delta_nps=delta_nps,
    )
    
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.plot(states[1], states[0])
    plt.show()
    np.save("Fossen_nps_zigzag_10_100_20_0.01.npy", states)
    print(states.shape)
    
    
    # zigzag 20
    # converged u under base thrust 50N: 0.659761
    time_step = 0.01
    
    init_psi = 0
    base_nps = 100
    delta_nps = 20.0
    zigzag_degrees = 20.0
    
    current_state = np.array([0, 0, init_psi, 0.659761, 0, 0]).reshape([6, 1])
    
    simulator = VesselSimulator(
        sample_hydro_params_2,
        time_step=time_step,
        model=Fossen,
        init_state=current_state,
    )
    thruster = NpsDoubleThruster(b=sample_b_2, **sample_thrust_params_2)
    
    states = fossen_nps_zigzag(
        simulator=simulator,
        thruster=thruster,
        current_state=current_state,
        zigzag_degrees=zigzag_degrees,
        base_nps=base_nps,
        delta_nps=delta_nps,
    )
    
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.plot(states[1], states[0])
    plt.show()
    
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(np.arange(states.shape[1])*time_step, states[2]/np.pi*180, label="$\psi$")
    axs[0].legend()
    axs[1].plot(np.arange(states.shape[1])*time_step, states[5], label="$r$")
    axs[1].legend()
    axs[2].plot(np.arange(states.shape[1])*time_step, states[3], label="$u$")
    axs[2].legend()
    axs[3].plot(np.arange(states.shape[1])*time_step, states[4], label="$v$")
    axs[3].legend()
    plt.show()
    
    np.save("Fossen_nps_zigzag_20_100_20_0.01.npy", states)
    print(states.shape)