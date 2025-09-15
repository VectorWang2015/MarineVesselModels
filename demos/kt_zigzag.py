import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.KT import FirstOrderResponse, sample_hydro_params, NoisyFirstOrderResponse


def kt_zigzag(
        simulator,
        zigzag_degrees: float,
        sample_delta: float,
        sample_u: float,
):
    init_psi = 0
    zigzag_psi = zigzag_degrees / 180 * np.pi
    
    current_state = np.array([0, 0, init_psi, sample_u, 0, 0]).reshape([6, 1])
    
    # turn right
    tgt_psi = init_psi + zigzag_psi
    tau = np.array([sample_delta]).reshape([1, 1])
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
    tau = np.array([-sample_delta]).reshape([1, 1])
    
    while current_state[2][0] > tgt_psi:
        current_state_with_tau = np.vstack((current_state, tau))
        states = np.hstack((states, current_state_with_tau))
    
        current_state = simulator.step(tau)
    
    # turn right
    tgt_psi = init_psi + zigzag_psi
    tau = np.array([sample_delta]).reshape([1, 1])
    
    while current_state[2][0] < tgt_psi:
        current_state_with_tau = np.vstack((current_state, tau))
        states = np.hstack((states, current_state_with_tau))
    
        current_state = simulator.step(tau)
    
    current_state_with_tau = np.vstack((current_state, np.zeros((1, 1))))
    states = np.hstack((states, current_state_with_tau))

    return states


if __name__ == "__main__":
    # zigzag 25/5
    """
    time_step = 0.01
    
    init_psi = 0
    zigzag_degrees = 25.0
    sample_delta = 5  # degrees
    sample_u = sample_hydro_params["u"]
    
    current_state = np.array([0, 0, init_psi, sample_u, 0, 0]).reshape([6, 1])
    
    simulator = VesselSimulator(
        sample_hydro_params,
        time_step=time_step,
        model=FirstOrderResponse,
        init_state=current_state,
    )
    
    # turning experiment
    # tgt_psi = init_psi + zigzag_degrees
    # tau = np.array([sample_delta]).reshape([1, 1])
    # states = None
    
    # for _ in range(3000):
    #     current_state_with_tau = np.vstack((current_state, tau))
    #     if states is not None:
    #         states = np.hstack((states, current_state_with_tau))
    #     else:
    #         states = current_state_with_tau.copy()
    
    #     current_state = simulator.step(tau)
    
    # print(current_state)
    
    states = kt_zigzag(
        simulator=simulator,
        zigzag_degrees=zigzag_degrees,
        sample_delta=sample_delta,
        sample_u=sample_u,
    )
    
    np.save("KT_zigzag_25_5.npy", states)
    
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.plot(states[1], states[0])
    plt.show()
    
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(np.arange(states.shape[1])*time_step, states[2]/np.pi*180, label="$\psi$")
    axs[0].legend()
    axs[1].plot(np.arange(states.shape[1])*time_step, states[5], label="$\omega$")
    axs[1].legend()
    axs[2].plot(np.arange(states.shape[1])*time_step, states[3], label="$u$")
    axs[2].legend()
    axs[3].plot(np.arange(states.shape[1])*time_step, states[4], label="$v$")
    axs[3].legend()
    plt.show()
    
    print(states.shape)
    print(states[:, -1])
    """
    
    
    # zigzag with white noise 25/5
    time_step = 0.01
    
    init_psi = 0
    zigzag_degrees = 25.0
    sample_delta = 5  # degrees
    sample_u = sample_hydro_params["u"]
    
    current_state = np.array([0, 0, init_psi, sample_u, 0, 0]).reshape([6, 1])
    
    simulator = VesselSimulator(
        sample_hydro_params,
        time_step=time_step,
        model=NoisyFirstOrderResponse,
        init_state=current_state,
    )
    
    states = kt_zigzag(
        simulator=simulator,
        zigzag_degrees=zigzag_degrees,
        sample_delta=sample_delta,
        sample_u=sample_u,
    )
    
    np.save("KT_noisy_zigzag_25_5.npy", states)
    
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
    
    print(states.shape)
    print(states[:, -1])
    