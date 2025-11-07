import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.thrusters import NpsDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_params_2, Fossen
from identification.PRBS import generate_dualthruster_prbs


def fossen_nps_prbs(
        simulator,
        thruster,
        current_state,
        l_seq,
        r_seq,
):
    states = None
    for F_l, F_r in zip(F_l_seq, F_r_seq):
        tau = thruster.newton_to_tau(F_l, F_r)
        current_state_with_tau = np.vstack((current_state, tau))
        states = current_state_with_tau if states is None else np.hstack((states, current_state_with_tau))
        
        current_state = simulator.step(tau)

    current_state_with_tau = np.vstack((current_state, tau))
    states = current_state_with_tau if states is None else np.hstack((states, current_state_with_tau))

    return states



if __name__ == "__main__":
    time_step = 0.1
    
    total_exp_time = 900    # seconds
    base_nps = 110
    delta_sum_nps = 60
    delta_diff_nps = 40
    
    current_state = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])

    t, nps_l, nps_r, *_ = generate_dualthruster_prbs(
        T_total=total_exp_time,
        dt=time_step,
        N0=base_nps,
        As=delta_sum_nps,
        Ad=delta_diff_nps,
        n_max=125,
        seed=114514,
    )

    # plt.plot(t, nps_l)
    # plt.plot(t, nps_r)
    # plt.show()
    simulator = VesselSimulator(
        sample_hydro_params_2,
        time_step=time_step,
        model=Fossen,
        init_state=current_state,
    )
    thruster = NpsDoubleThruster(b=sample_b_2, **sample_thrust_params_2)

    # check & plot F_x, N_r
    taus = None
    for l, r in zip(nps_l, nps_r):
        tau = thruster.n_to_tau(l, r, u=0)
        taus = tau if taus is None else np.hstack((taus, tau))
    plt.plot(np.arange(taus.shape[1]), taus[0], label="F_x")
    plt.plot(np.arange(taus.shape[1]), taus[2], label="N_r")
    plt.legend()
    plt.show()
    
    """
    states = fossen_prbs(
        simulator=simulator,
        thruster=thruster,
        current_state=current_state,
        F_l_seq=F_l,
        F_r_seq=F_r,
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
    
    np.save("Fossen_PRBS_0.1_900.npy", states)
    print(states.shape)
    """