import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import NoisyVesselSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen
from MarineVesselModels.noises import GaussMarkovNoiseGenerator

from control.pid import PID
from control.lqr import HeadingLQR
from control.los import LOSGuider


if __name__ == "__main__":
    time_step = 0.1
    total_exp_steps = 2000
    control_step = 1
    control_every = int(control_step / time_step)
    desire_psi = np.pi / 2

    base_N = 100.0
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

    # diff_controller = PID(
    #     kp=50,
    #     ki=5,
    #     kd=25,
    #     buffer_size=10,
    #     min_saturation=-max_diff_N/2,
    #     max_saturation=max_diff_N/2,
    # )

    Q = np.array([[15, 0], [0, 1]])
    R = np.array([[0.03]])
    diff_controller = HeadingLQR(
        m33=sample_hydro_params_2["m33"],
        d33=sample_hydro_params_2["N_r"],
        b=sample_b_2,
        Q=Q,
        R=R,
    )
    guider = LOSGuider(waypoints=waypoints, reached_threshold=2)

    ts = []
    psis = []
    psi_errs = []
    lefts = []
    rights = []
    xs = []
    ys = []
    us = []
    vs = []
    rs = []
    is_ended = False
    
    for t in range(total_exp_steps):
        current_x = current_state[0][0]
        current_y = current_state[1][0]
        current_psi = current_state[2][0]
        current_u = current_state[3][0]
        current_v = current_state[4][0]
        current_r = current_state[5][0]

        # control every control_step seconds
        if t % control_every == 0:
            is_ended, psi_err = guider.step((current_x, current_y), current_psi)
            # calculate new tau
            #control_signal = diff_controller.control(error=psi_err)
            control_signal = diff_controller.control(error_delta=psi_err, r=current_r)
            # truncate
            control_signal = max_diff_N/2 if control_signal > max_diff_N/2 else control_signal
            control_signal = -max_diff_N/2 if control_signal < -max_diff_N/2 else control_signal

            left = base_N + control_signal
            right = base_N - control_signal
            tau = thruster.newton_to_tau(left, right)

        # record
        ts.append(t)
        psis.append(current_psi/np.pi*180)
        psi_errs.append(psi_err/np.pi*180)
        lefts.append(left)
        rights.append(right)
        xs.append(current_x)
        ys.append(current_y)
        us.append(current_u)
        vs.append(current_v)
        rs.append(current_r)

        if is_ended:
            break

        # apply tau and step
        current_state = simulator.step(tau)
    
    way_xs = [pt[0] for pt in waypoints]
    way_ys = [pt[1] for pt in waypoints]

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.scatter(way_ys, way_xs, marker="x", label="Desired waypoints", color="red")
    ax.plot(ys, xs, label="Actual trajectory")
    ax.legend()
    plt.show()

    ts = np.array(ts)

    fig, axs = plt.subplots(2, 2)
    axs[0][0].plot(ts*time_step, psis, label="$\psi$")
    axs[0][0].plot(ts*time_step, psi_errs, label="$\psi$ error", color="r")
    axs[0][0].legend()
    axs[0][0].set_ylabel("angle in $^\circ$")
    axs[0][0].set_xlabel("time in $s$")
    axs[1][0].plot(ts*time_step, lefts, label="left thrust")
    axs[1][0].plot(ts*time_step, rights, label="right thrust")
    axs[1][0].set_ylabel("force in N")
    axs[1][0].set_xlabel("time in $s$")
    axs[1][0].legend()
    axs[0][1].plot(ts*time_step, us, label="$u$")
    axs[0][1].plot(ts*time_step, vs, label="$v$")
    axs[0][1].set_ylabel("speed in $m\cdot s^{-1}$")
    axs[0][1].set_xlabel("time in $s$")
    axs[0][1].legend()
    axs[1][1].plot(ts*time_step, rs, label="$r$")
    axs[1][1].set_ylabel("angular velocity in $rad\cdot s^{-1}$")
    axs[1][1].set_xlabel("time in $s$")
    axs[1][1].legend()

    plt.show()
