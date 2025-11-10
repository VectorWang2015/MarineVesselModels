import numpy as np
from matplotlib import pyplot as plt
from json import dumps

from MarineVesselModels.simulator import VesselSimulator, NoisyVesselSimulator
from MarineVesselModels.thrusters import NpsDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_params_2, Fossen
from identification.PRBS import generate_dualthruster_prbs
from MarineVesselModels.noises import GaussianNoiseGenerator, GaussMarkovNoiseGenerator


def fossen_nps_prbs(
        simulator,
        thruster: NpsDoubleThruster,
        current_state,
        l_seq,
        r_seq,
):
    states = None
    for rps_l, rps_r in zip(l_seq, r_seq):
        # use rps and current u as input, to compute tau
        u = current_state[3][0]
        tau = thruster.n_to_tau(l_nps=rps_l, r_nps=rps_r, u=u)
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
    
    current_obsv = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])

    t, nps_l, nps_r, *_ = generate_dualthruster_prbs(
        T_total=total_exp_time,
        dt=time_step,
        N0=base_nps,
        As=delta_sum_nps,
        Ad=delta_diff_nps,
        n_max=125,
        seed=114514,
    )

    # setup noise simuls
    # std dev for x, y, psi, u, v, r
    sigmas = np.array([0.05, 0.05, 1/180*3.1415, 0.01, 0.01, 0.005])
    Sigma = np.diag(sigmas**2)
    obsv_noise_gen = GaussianNoiseGenerator(Sigma=Sigma)
    #obsv_noise_gen = GaussianNoiseGenerator()
    tau_vec = np.array([60.0, 60.0, 30.0])  # s for X,Y,N
    sigmas = np.array([5.0, 5.0, 1.0])     # N, N, N·m (steady-obsv std)
    Sigma = np.diag(sigmas**2)
    tau_noise_gen = GaussMarkovNoiseGenerator(dt=time_step, Sigma=Sigma, tau=tau_vec)

    simulator = NoisyVesselSimulator(
        sample_hydro_params_2,
        time_step=time_step,
        model=Fossen,
        init_state=current_obsv,
        obsv_noise_gen=obsv_noise_gen,
        tau_noise_gen=tau_noise_gen,
    )

    """
    simulator = VesselSimulator(
        sample_hydro_params_2,
        time_step=time_step,
        model=Fossen,
        init_state=current_obsv,
    )
    """
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

    
    states = fossen_nps_prbs(
        simulator=simulator,
        thruster=thruster,
        current_state=current_obsv,
        l_seq=nps_l,
        r_seq=nps_r,
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
    
    #np.save("Fossen_PRBS_0.1_900.npy", states)
    print(states.shape)
    xs = list(states[0])
    ys = list(states[1])
    psis = list(states[2])
    us = list(states[3])
    vs = list(states[4])
    rs = list(states[5])
    Fxs = list(states[6])
    Fys = list(states[7])
    Nrs = list(states[8])
    nps_l = list(nps_l)
    nps_r = list(nps_r)

    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    psis = [float(psi) for psi in psis]
    us = [float(u) for u in us]
    vs = [float(v) for v in vs]
    rs = [float(r) for r in rs]
    Fxs = [float(Fx) for Fx in Fxs]
    Fys = [float(Fy) for Fy in Fys]
    Nrs = [float(Nr) for Nr in Nrs]
    nps_l = [float(l) for l in nps_l]
    nps_r = [float(r) for r in nps_r]

    exp_record = {
        "xs": xs,
        "ys": ys,
        "psis": psis,
        "us": us,
        "vs": vs,
        "rs": rs,
        "Fxs": Fxs,
        "Fys": Fys,
        "Nrs": Nrs,
        "rps_l": nps_l,
        "rps_r": nps_r,
    }

    with open("noised_Fossen_PRBS_nps_0.1_900.json", "w") as fd:
        fd.write(dumps(exp_record))