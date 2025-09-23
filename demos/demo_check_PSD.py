import numpy as np
from identification.PRBS import generate_dualthruster_prbs, plot_PSD
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.Fossen import sample_hydro_params_2, sample_b_2, Fossen
from MarineVesselModels.thrusters import NaiveDoubleThruster

def fossen_zigzag(
        simulator,
        thruster,
        current_state,
        zigzag_degrees: float,
        base_N: float,
        base_delta_N: float,
):
    init_psi = 0
    current_state = current_state.copy()
    zigzag_psi = zigzag_degrees/180*np.pi

    F_ls = []
    F_rs = []
    taus = []

    # turn right
    tgt_psi = init_psi + zigzag_psi
    left_thr = base_N + base_delta_N
    right_thr = base_N - base_delta_N
    tau = thruster.newton_to_tau(left_thr, right_thr)
    
    while current_state[2][0] < tgt_psi:
        F_ls.append(left_thr)
        F_rs.append(right_thr)
        taus.append(tau)
        current_state = simulator.step(tau)
    
    # turn left
    tgt_psi = init_psi - zigzag_psi
    left_thr = base_N - base_delta_N
    right_thr = base_N + base_delta_N
    tau = thruster.newton_to_tau(left_thr, right_thr)
    
    while current_state[2][0] > tgt_psi:
        F_ls.append(left_thr)
        F_rs.append(right_thr)
        taus.append(tau)
        current_state = simulator.step(tau)
    
    return F_ls, F_rs, np.hstack(taus)

if __name__ == "__main__":
    dt = 0.1
    init_psi = 0
    base_N = 50.0
    base_delta_N = 20.0
    zigzag_degrees = 10.0
    
    current_state = np.array([0, 0, init_psi, 0.659761, 0, 0]).reshape([6, 1])
    
    simulator = VesselSimulator(
        sample_hydro_params_2,
        time_step=dt,
        model=Fossen,
        init_state=current_state,
    )
    thruster = NaiveDoubleThruster(b=sample_b_2)
    
    F_ls, F_rs, taus = fossen_zigzag(
        simulator=simulator,
        thruster=thruster,
        current_state=current_state,
        zigzag_degrees=zigzag_degrees,
        base_N=base_N,
        base_delta_N=base_delta_N,
    )

    tau1 = list(taus[0])
    tau3 = list(taus[2])
    t, n_L, n_R, n_sum, n_delta = generate_dualthruster_prbs(1200, dt)

    fig, axs = plt.subplots(2, 2)
    tau1 = tau1*300
    tau3 = tau3*300

    axs[0][0].plot(t[:len(tau1)], n_sum[:len(tau1)])
    axs[0][0].plot(t[:len(tau1)], tau1)
    axs[0][0].set_ylabel("Fx")

    axs[0][1].plot(t[:len(tau3)], n_delta[:len(tau3)])
    axs[0][1].plot(t[:len(tau3)], tau3)
    axs[0][1].set_ylabel("Nr")


    n_sum = n_sum[:len(tau1)]
    n_delta = n_delta[:len(tau1)]

    plot_PSD(axs[1][0], n_sum, 1/dt)
    plot_PSD(axs[1][0], tau1, 1/dt)

    plot_PSD(axs[1][1], n_delta, 1/dt)
    plot_PSD(axs[1][1], tau3, 1/dt)
    plt.show()