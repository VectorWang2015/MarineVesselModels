import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import NoisyVesselSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen
from MarineVesselModels.noises import GaussMarkovNoiseGenerator

from control.pid import DoubleLoopHeadingPID


if __name__ == "__main__":
    # single loop pid
    """
    time_step = 0.1
    total_control_steps = 200
    control_step = 1
    control_every = int(control_step / time_step)
    desire_psi = np.pi / 2

    base_N = 0.0
    max_diff_N = 60.0   # max diff between thrusters
    
    current_state = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])

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
    diff_controller = PID(
        kp=50,
        ki=5,
        kd=25,
        buffer_size=10,
        min_saturation=-max_diff_N/2,
        max_saturation=max_diff_N/2,
    )

    ts = []
    psis = []
    psi_errs = []
    lefts = []
    rights = []
    
    for t in range(total_control_steps):
        current_psi = current_state[2][0]
        psi_err = desire_psi - current_psi
        psi_err %= 2*np.pi
        psi_err = psi_err - np.pi*2 if psi_err > np.pi else psi_err

        # control every control_step seconds
        if t % control_every == 0:
            # calculate new tau
            control_signal = diff_controller.control(error=psi_err)

            left = base_N + control_signal
            right = base_N - control_signal
            tau = thruster.newton_to_tau(left, right)

        # record
        ts.append(t)
        psis.append(current_psi/np.pi*180)
        psi_errs.append(psi_err/np.pi*180)
        lefts.append(left)
        rights.append(right)

        # apply tau and step
        current_state = simulator.step(tau)

    ts = np.array(ts) * time_step
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(ts, psis, label="$\psi$")
    axs[0].plot(ts, psi_errs, label="$\psi$ error", color="r")
    axs[0].legend()
    axs[1].plot(ts, lefts, label="left thrust")
    axs[1].plot(ts, rights, label="right thrust")
    axs[1].legend()

    plt.show()
    """

    # double-loop pid
    time_step = 0.1
    total_control_steps = 200
    control_step = 0.2
    control_every = int(control_step / time_step)
    desire_psi = np.pi / 2
    test_desire_r = 0.4

    base_N = 0.0
    max_diff_N = 60.0   # max diff between thrusters
    
    current_state = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])

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
    # control inner loop, r as target
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

    ts = []
    psis = []
    psi_errs = []
    rs = []
    ref_rs = []
    lefts = []
    rights = []
    
    for t in range(total_control_steps):
        current_psi = current_state[2][0]
        current_r = current_state[5][0]
        psi_err = desire_psi - current_psi
        psi_err %= 2*np.pi
        psi_err = psi_err - np.pi*2 if psi_err > np.pi else psi_err
        r_err = test_desire_r - current_r

        # control every control_step seconds
        if t % control_every == 0:
            # calculate new tau
            control_signal, r_ref = diff_controller.step(psi_ref=desire_psi, r=current_r, psi=current_psi)

            left = base_N + control_signal
            right = base_N - control_signal
            tau = thruster.newton_to_tau(left, right)

        # record
        ts.append(t)
        psis.append(current_psi/np.pi*180)
        psi_errs.append(psi_err/np.pi*180)
        rs.append(current_r)
        ref_rs.append(r_ref)
        lefts.append(left)
        rights.append(right)

        # apply tau and step
        current_state = simulator.step(tau)

    ts = np.array(ts) * time_step
    fig, axs = plt.subplots(3, 1)
    #axs[0].plot(ts, psis, label="$\psi$")
    axs[0].plot(ts, ref_rs, label="ref $r$")
    axs[0].plot(ts, rs, label="$r$", color="black")
    axs[0].legend()
    axs[1].plot(ts, psis, label="$\psi$", color="black")
    axs[1].plot(ts, psi_errs, label="$\psi$ error", color="r")
    axs[1].legend()
    axs[2].plot(ts, lefts, label="left thrust")
    axs[2].plot(ts, rights, label="right thrust")
    axs[2].legend()

    plt.show()