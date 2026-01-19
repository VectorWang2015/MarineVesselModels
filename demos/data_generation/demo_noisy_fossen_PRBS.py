import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import NoisyVesselSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen
from MarineVesselModels.noises import GaussianNoiseGenerator, GaussMarkovNoiseGenerator
from identification.PRBS import generate_dualthruster_prbs

from pickle import dump

def fossen_prbs(
        simulator,
        thruster,
        current_obsv,
        F_l_seq,
        F_r_seq,
):
    obsvs = None
    for F_l, F_r in zip(F_l_seq, F_r_seq):
        tau = thruster.newton_to_tau(F_l, F_r)
        current_obsv_with_tau = np.vstack((current_obsv, tau))
        obsvs = current_obsv_with_tau if obsvs is None else np.hstack((obsvs, current_obsv_with_tau))
        
        current_obsv = simulator.step(tau)

    current_obsv_with_tau = np.vstack((current_obsv, tau))
    obsvs = current_obsv_with_tau if obsvs is None else np.hstack((obsvs, current_obsv_with_tau))

    return obsvs



if __name__ == "__main__":
    time_step = 0.1
    
    total_exp_time = 900    # seconds
    base_N = 50.0
    delta_sum_N = 30.0
    delta_diff_N = 20.0
    
    current_obsv = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])

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

    t, F_l, F_r, *_ = generate_dualthruster_prbs(
        T_total=total_exp_time,
        dt=time_step,
        N0=base_N,
        As=delta_sum_N,
        Ad=delta_diff_N,
        n_max=125,
        seed=114514,
    )

    # plt.plot(t, F_l)
    # plt.plot(t, F_r)
    # plt.show()
    
    simulator = NoisyVesselSimulator(
        sample_hydro_params_2,
        time_step=time_step,
        model=Fossen,
        init_state=current_obsv,
        obsv_noise_gen=obsv_noise_gen,
        tau_noise_gen=tau_noise_gen,
    )
    thruster = NaiveDoubleThruster(b=sample_b_2)
    
    obsvs = fossen_prbs(
        simulator=simulator,
        thruster=thruster,
        current_obsv=current_obsv,
        F_l_seq=F_l,
        F_r_seq=F_r,
    )
    
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.plot(obsvs[1], obsvs[0])
    plt.show()
    
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(np.arange(obsvs.shape[1])*time_step, obsvs[2]/np.pi*180, label="$\psi$")
    axs[0].legend()
    axs[1].plot(np.arange(obsvs.shape[1])*time_step, obsvs[5], label="$r$")
    axs[1].legend()
    axs[2].plot(np.arange(obsvs.shape[1])*time_step, obsvs[3], label="$u$")
    axs[2].legend()
    axs[3].plot(np.arange(obsvs.shape[1])*time_step, obsvs[4], label="$v$")
    axs[3].legend()
    plt.show()
    
    np.save("noised_Fossen_PRBS_0.1_900.npy", obsvs)
    print(obsvs.shape)

    file_handler = open("noised_Fossen_PRBS_0.1_900.pickle.obj", "wb")
    dump(simulator.summarize(), file_handler)