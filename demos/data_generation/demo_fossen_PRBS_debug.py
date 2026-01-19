import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen
from identification.PRBS import generate_dualthruster_prbs


def fossen_prbs(
        simulator,
        thruster,
        current_state,
        F_l_seq,
        F_r_seq,
):
    states = None
    partial_states = None
    for F_l, F_r in zip(F_l_seq, F_r_seq):
        tau = thruster.newton_to_tau(F_l, F_r)
        current_state_with_tau = np.vstack((current_state, tau))
        states = current_state_with_tau if states is None else np.hstack((states, current_state_with_tau))
        
        current_state, current_partial_state = simulator.step(tau)
        # state is one-step ahead of partial_state
        partial_states = current_partial_state if partial_states is None else np.hstack((partial_states, current_partial_state))

    current_state_with_tau = np.vstack((current_state, tau))
    states = np.hstack((states, current_state_with_tau))

    return states, partial_states



if __name__ == "__main__":
    time_step = 0.1
    
    total_exp_time = 900    # seconds
    base_N = 50.0
    delta_sum_N = 30.0
    delta_diff_N = 20.0

    seed = 114514
    
    current_state = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])

    t, F_l, F_r, *_ = generate_dualthruster_prbs(
        T_total=total_exp_time,
        dt=time_step,
        N0=base_N,
        As=delta_sum_N,
        Ad=delta_diff_N,
        n_max=125,
        seed=seed,
    )

    # plt.plot(t, F_l)
    # plt.plot(t, F_r)
    # plt.show()
    
    simulator = VesselSimulator(
        sample_hydro_params_2,
        time_step=time_step,
        model=Fossen,
        init_state=current_state,
        debug=True,
    )
    thruster = NaiveDoubleThruster(b=sample_b_2)
    
    states, partial_states = fossen_prbs(
        simulator=simulator,
        thruster=thruster,
        current_state=current_state,
        F_l_seq=F_l,
        F_r_seq=F_r,
    )

    print(states.shape)
    print(partial_states.shape)
    
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
    
    np.save("Fossen_PRBS_0.1_900.npy", states)
    np.save("partial_Fossen_PRBS_0.1_900.npy", partial_states)