import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import VesselSimulator, NoisyVesselSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen
from MarineVesselModels.noises import GaussianNoiseGenerator, GaussMarkovNoiseGenerator


def fossen_zigzag(
        simulator,
        thruster,
        current_obsv,
        zigzag_degrees: float,
        base_N: float,
        base_delta_N: float,
):
    init_psi = 0
    current_obsv = current_obsv.copy()
    zigzag_psi = zigzag_degrees/180*np.pi

    # turn right
    tgt_psi = init_psi + zigzag_psi
    left_thr = base_N + base_delta_N
    right_thr = base_N - base_delta_N
    tau = thruster.newton_to_tau(left_thr, right_thr)
    obsvs = None
    
    while current_obsv[2][0] < tgt_psi:
        current_obsv_with_tau = np.vstack((current_obsv, tau))
        if obsvs is not None:
            obsvs = np.hstack((obsvs, current_obsv_with_tau))
        else:
            obsvs = current_obsv_with_tau.copy()
    
        current_obsv = simulator.step(tau)
    
    # turn left
    tgt_psi = init_psi - zigzag_psi
    left_thr = base_N - base_delta_N
    right_thr = base_N + base_delta_N
    tau = thruster.newton_to_tau(left_thr, right_thr)
    
    while current_obsv[2][0] > tgt_psi:
        current_obsv_with_tau = np.vstack((current_obsv, tau))
        obsvs = np.hstack((obsvs, current_obsv_with_tau))
    
        current_obsv = simulator.step(tau)
    
    # turn right
    tgt_psi = init_psi + zigzag_psi
    left_thr = base_N + base_delta_N
    right_thr = base_N - base_delta_N
    tau = thruster.newton_to_tau(left_thr, right_thr)
    
    while current_obsv[2][0] < tgt_psi:
        current_obsv_with_tau = np.vstack((current_obsv, tau))
        obsvs = np.hstack((obsvs, current_obsv_with_tau))
    
        current_obsv = simulator.step(tau)
    
    current_obsv_with_tau = np.vstack((current_obsv, tau))
    obsvs = np.hstack((obsvs, current_obsv_with_tau))

    return obsvs


if __name__ == "__main__":
    # zigzag 20
    # converged u under base thrust 50N: 0.659761
    time_step = 0.01
    
    init_psi = 0
    base_N = 50.0
    base_delta_N = 20.0
    zigzag_degrees = 20.0
    
    current_obsv = np.array([0, 0, init_psi, 0.659761, 0, 0]).reshape([6, 1])

    # setup noise simuls
    # std dev for x, y, psi, u, v, r
    sigmas = np.array([0.05, 0.05, 1/180*3.1415, 0.01, 0.01, 0.005])
    Sigma = np.diag(sigmas**2)
    obsv_noise_gen = GaussianNoiseGenerator(Sigma=Sigma)
    #obsv_noise_gen = GaussianNoiseGenerator()
    tau_vec = np.array([60.0, 60.0, 30.0])  # s for X,Y,N
    sigmas = np.array([5.0, 5.0, 1.0])     # N, N, N·m (steady-state std)
    Sigma = np.diag(sigmas**2)
    tau_noise_gen = GaussMarkovNoiseGenerator(dt=time_step, Sigma=Sigma, tau=tau_vec)
        
    
    n_simulator = NoisyVesselSimulator(
        sample_hydro_params_2,
        time_step=time_step,
        model=Fossen,
        init_state=current_obsv,
        tau_noise_gen=tau_noise_gen,
        obsv_noise_gen=obsv_noise_gen,
    )
    thruster = NaiveDoubleThruster(b=sample_b_2)
    
    plain_simulator = NoisyVesselSimulator(
        sample_hydro_params_2,
        time_step=time_step,
        model=Fossen,
        init_state=current_obsv,
    )
    
    n_obsvs = fossen_zigzag(
        simulator=n_simulator,
        thruster=thruster,
        current_obsv=current_obsv,
        zigzag_degrees=zigzag_degrees,
        base_N=base_N,
        base_delta_N=base_delta_N,
    )

    # observations of no-noise simulation
    obsvs = fossen_zigzag(
        simulator=plain_simulator,
        thruster=thruster,
        current_obsv=current_obsv,
        zigzag_degrees=zigzag_degrees,
        base_N=base_N,
        base_delta_N=base_delta_N,
    )
    
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.plot(obsvs[1], obsvs[0])
    ax.plot(n_obsvs[1], n_obsvs[0])
    plt.show()
    
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(np.arange(obsvs.shape[1])*time_step, obsvs[2]/np.pi*180, label="$\psi$")
    axs[0].plot(np.arange(n_obsvs.shape[1])*time_step, n_obsvs[2]/np.pi*180, label="noised $\psi$")
    axs[0].legend()
    axs[1].plot(np.arange(obsvs.shape[1])*time_step, obsvs[5], label="$r$")
    axs[1].plot(np.arange(n_obsvs.shape[1])*time_step, n_obsvs[5], label="noised $r$")
    axs[1].legend()
    axs[2].plot(np.arange(obsvs.shape[1])*time_step, obsvs[3], label="$u$")
    axs[2].plot(np.arange(n_obsvs.shape[1])*time_step, n_obsvs[3], label="noised $u$")
    axs[2].legend()
    axs[3].plot(np.arange(obsvs.shape[1])*time_step, obsvs[4], label="$v$")
    axs[3].plot(np.arange(n_obsvs.shape[1])*time_step, n_obsvs[4], label="noised $v$")
    axs[3].legend()
    plt.show()
    
    #np.save("Fossen_zigzag_20_50_20_0.01.npy", obsvs)
    print(obsvs.shape)