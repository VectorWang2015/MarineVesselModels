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

    # turn right
    tgt_psi = init_psi + zigzag_psi
    left_thr = base_N + base_delta_N
    right_thr = base_N - base_delta_N
    tau = thruster.newton_to_tau(left_thr, right_thr)
    
    while current_state[2][0] < tgt_psi:
        F_ls.append(left_thr)
        F_rs.append(right_thr)
        current_state = simulator.step(tau)
    
    # turn left
    tgt_psi = init_psi - zigzag_psi
    left_thr = base_N - base_delta_N
    right_thr = base_N + base_delta_N
    tau = thruster.newton_to_tau(left_thr, right_thr)
    
    while current_state[2][0] > tgt_psi:
        F_ls.append(left_thr)
        F_rs.append(right_thr)
        current_state = simulator.step(tau)
    
    return F_ls, F_rs

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
    
    F_ls, F_rs = fossen_zigzag(
        simulator=simulator,
        thruster=thruster,
        current_state=current_state,
        zigzag_degrees=zigzag_degrees,
        base_N=base_N,
        base_delta_N=base_delta_N,
    )

    t, n_L, n_R, n_sum, n_delta = generate_dualthruster_prbs(1200, dt)

    F_ls = F_ls*300
    print(len(n_L))
    print(len(F_ls))

    plt.plot(t[:len(F_ls)], n_L[:len(F_ls)])
    plt.plot(t[:len(F_ls)], F_ls)
    plt.show()

    n_L = n_L[:len(F_ls)]
    fig, ax = plt.subplots()
    plot_PSD(ax, n_L, 1/dt)
    plot_PSD(ax, F_ls, 1/dt)
    plt.show()