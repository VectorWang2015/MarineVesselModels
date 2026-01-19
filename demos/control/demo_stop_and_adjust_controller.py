import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import NoisyVesselSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen
from MarineVesselModels.noises import GaussMarkovNoiseGenerator

from control.pid import DoubleLoopHeadingPID, PIDAW
from control.los import LOSGuider
from control.controller import DualThrustUSVStopAndAdjustController


if __name__ == "__main__":
    # case 2, PID for stable u, Double-loop for heading control
    time_step = 0.1
    total_exp_steps = 2000
    control_step = 0.2
    control_every = int(control_step / time_step)

    psi_threshold = 30 / 180 * np.pi
    u_cruise = 1.0
    max_base_N = 100.0
    max_diff_N = 60.0   # max diff between thrusters

    waypoints = [
        (20, 5),
        (0, 10),
        (15, 15),
        (0, 20),
        (10, 25),
        (0, 30),
        (5, 35),
        (0, 40),
    ]
    
    current_state = np.array([0, 0, -np.pi, 0, 0, 0]).reshape([6, 1])

    # setup noise simuls
    # std dev for x, y, psi, u, v, r
    # sigmas = np.array([0.05, 0.05, 1/180*3.1415, 0.01, 0.01, 0.005])
    # Sigma = np.diag(sigmas**2)
    # obsv_noise_gen = GaussianNoiseGenerator(Sigma=Sigma)
    #obsv_noise_gen = GaussianNoiseGenerator()
    tau_vec = np.array([60.0, 60.0, 30.0])  # s for X,Y,N
    sigmas = np.array([5.0, 5.0, 1.0])     # N, N, N·m (steady-obsv std)
    Sigma = np.diag(sigmas**2)
    tau_noise_gen = GaussMarkovNoiseGenerator(dt=time_step, Sigma=Sigma, tau=tau_vec)

    simulator = NoisyVesselSimulator(
        sample_hydro_params_2,
        time_step=time_step,
        model=Fossen,
        init_state=current_state,
        tau_noise_gen=tau_noise_gen,
    )
    thruster = NaiveDoubleThruster(b=sample_b_2)

    diff_controller = DoubleLoopHeadingPID(
                dt=control_step,
                # outer loop (psi -> r)
                psi_kp=1, psi_ki=0.2, psi_kd=0.05,
                r_ref_lim=0.45,           # staturation for r (rad/s)
                r_kp=200, r_ki=250, r_kd=0,
                u_lim=max_diff_N/2,               # satuaration for control value
                r_ref_slew=None,     # slew rate for r (rad/s^2)
                u_slew=None,         # slew rate for u
    )
    u_controller = PIDAW(
        kp=150,
        ki=200,
        kd=5,
        dt=control_step,
        u_min=-max_base_N,
        u_max=max_base_N,
    )
    guider = LOSGuider(waypoints=waypoints, reached_threshold=1, output_err_flag=False)

    controller = DualThrustUSVStopAndAdjustController(
        psi_threshold=psi_threshold,
        u_cruise=u_cruise,
        psi_controller=diff_controller,
        u_controller=u_controller,
        guider=guider,
    )

    is_ended = False

    while not is_ended:
        control_cmd = controller.step(current_state)
        if control_cmd is None:
            is_ended = False
            break

        left, right = control_cmd
        tau = thruster.newton_to_tau(l_thrust_N=left, r_thrust_N=right)
        current_state = simulator.step(tau)

    result = controller.summarize()

    xs = result["xs"]
    ys = result["ys"]
    psis = (result["psis"] + np.pi) % (2*np.pi) - np.pi
    us = result["us"]
    vs = result["vs"]
    rs = result["rs"]
    desired_us = result["desired_us"]
    desired_rs = result["desired_rs"]
    desired_psis = (result["desired_psis"] + np.pi) % (2*np.pi) - np.pi
    u_errs = desired_us - us
    psi_errs = desired_psis - psis
    r_errs = desired_rs - rs
    lefts = result["lefts"]
    rights = result["rights"]

    way_xs = [pt[0] for pt in waypoints]
    way_ys = [pt[1] for pt in waypoints]

    ts = np.arange(len(xs)) * time_step

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.scatter(way_ys, way_xs, marker="x", label="Desired waypoints", color="red")
    ax.plot(ys, xs, label="Actual trajectory")
    ax.legend()
    plt.show()

    fig, axs = plt.subplots(2, 2)
    axs[0][0].plot(ts, psis/np.pi*180, label="$\psi$")
    #axs[0][0].plot(ts, psi_errs, label="$\psi$ error", color="r")
    axs[0][0].plot(ts, desired_psis/np.pi*180, label="desired $\psi$", color="r")
    axs[0][0].legend()
    axs[0][0].set_ylabel("angle in $^\circ$")
    axs[0][0].set_xlabel("time in $s$")
    axs[1][0].plot(ts, lefts, label="left thrust")
    axs[1][0].plot(ts, rights, label="right thrust")
    axs[1][0].set_ylabel("force in N")
    axs[1][0].set_xlabel("time in $s$")
    axs[1][0].legend()

    axs[1][1].plot(ts, us, label="$u$")
    axs[1][1].plot(ts, desired_us, label="desired $u$", color="r")
    axs[1][1].plot(ts, vs, label="$v$")
    axs[1][1].set_ylabel("speed in $m\cdot s^{-1}$")
    axs[1][1].set_xlabel("time in $s$")
    axs[1][1].legend()
    
    axs[0][1].plot(ts, rs, label="$r$")
    axs[0][1].plot(ts, desired_rs, label="desired $r$", color="r")
    axs[0][1].set_ylabel("angular velocity in $rad\cdot s^{-1}$")
    axs[0][1].set_xlabel("time in $s$")
    axs[0][1].legend()

    plt.show()