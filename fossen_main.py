import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import FossenSimulator
from MarineVesselModels.stepper import time_invariant_simulator_runge_kutta, time_invariant_simulator_euler
from MarineVesselModels.thrusters import NaiveDoubleThruster

if __name__ == "__main__":

    # data from <双体双推进无人船路径跟踪控制研究>
    hydro_params = {
        "d11": 62.5,
        "d22": 0.0,
        "d33": 79.5,
        "m": 100.0,
        "X_dotu": -20.0,
        "Y_dotv": -21.6,
        "N_dotr": -206.2,
        "I": 20.0,
    }
    # data from <Research on Parameter Identification Method of Four-Thrusters AUSV Dynamics Model>
    b_2 = 0.5
    hydro_params_2 = {
        "d11": 6.0,
        "d22": 7.1,
        "d33": 0.8,
        "m11": 13.0,
        "m22": 23.3,
        "m33": 1.3,
    }

    step_lim = 200
    t = 0.1

    """
    # scene 1: rotation for hydro setup1
    # diameter should be around 12, for hydro 1
    tau = np.array([9.46*3.75, 0, 5.68*(1.25-2.5)]).reshape([3, 1])
    

    # calculate with runge-kutta
    simulator_rk = LinearFossenSimulator(
        hydro_params, time_step=t, step_fn=time_invariant_simulator_runge_kutta)
    rk_xs = []
    rk_ys = []

    for _ in range(step_lim):
        state = simulator_rk.step(tau)
        #print(state)
        x = state[0][0]
        y = state[1][0]
        psi = state[2][0]
        u = state[3][0]
        v = state[4][0]
        r = state[5][0]

        rk_xs.append(x)
        rk_ys.append(y)


    # calculate with euler
    simulator_e = LinearFossenSimulator(
        hydro_params, time_step=t, step_fn=time_invariant_simulator_euler)
    e_xs = []
    e_ys = []

    for _ in range(step_lim):
        state = simulator_e.step(tau)
        #print(state)
        x = state[0][0]
        y = state[1][0]
        psi = state[2][0]
        u = state[3][0]
        v = state[4][0]
        r = state[5][0]

        e_xs.append(x)
        e_ys.append(y)


    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    ax.plot(rk_ys, rk_xs, label="Runge Kutta")
    ax.plot(e_ys, e_xs, label="Euler")
    ax.legend()

    plt.show()
    """

    # scene 2: rotation for hydro setup2
    thruster = NaiveDoubleThruster(b=b_2)
    tau = thruster.newton_to_tau(
        l_thrust_N=-1.0,
        r_thrust_N=2.5,
    )
    

    # calculate with runge-kutta
    simulator_rk = FossenSimulator(
        hydro_params_2, time_step=t, step_fn=time_invariant_simulator_runge_kutta)
    rk_xs = []
    rk_ys = []

    for _ in range(step_lim):
        state = simulator_rk.step(tau)
        #print(state)
        x = state[0][0]
        y = state[1][0]
        psi = state[2][0]
        u = state[3][0]
        v = state[4][0]
        r = state[5][0]

        rk_xs.append(x)
        rk_ys.append(y)


    # calculate with euler
    simulator_e = FossenSimulator(
        hydro_params_2, time_step=t, step_fn=time_invariant_simulator_euler)
    e_xs = []
    e_ys = []

    for _ in range(step_lim):
        state = simulator_e.step(tau)
        #print(state)
        x = state[0][0]
        y = state[1][0]
        psi = state[2][0]
        u = state[3][0]
        v = state[4][0]
        r = state[5][0]

        e_xs.append(x)
        e_ys.append(y)


    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    ax.plot(rk_ys, rk_xs, label="Runge Kutta")
    ax.plot(e_ys, e_xs, label="Euler")
    ax.legend()

    plt.show()