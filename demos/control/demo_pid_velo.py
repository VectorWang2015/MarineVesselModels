import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import NoisyVesselSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen
from MarineVesselModels.noises import GaussMarkovNoiseGenerator

from control.pid import PIDAW

def u_experiment(
            time_step,
            total_control_steps,
            control_step,
            desire_u,
            max_base_N,
            kp, ki, kd,
):
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

    control_every = int(control_step / time_step)
    current_state = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])

    simulator = NoisyVesselSimulator(
        sample_hydro_params_2,
        time_step=time_step,
        model=Fossen,
        init_state=current_state,
        tau_noise_gen=tau_noise_gen,
    )
    thruster = NaiveDoubleThruster(b=sample_b_2)
    u_controller = PIDAW(
        kp=kp,
        ki=ki,
        kd=kd,
        dt=control_step,
        u_min=-max_base_N,
        u_max=max_base_N,
    )

    ts = []
    us = []
    u_errs = []
    control_signals = []
    
    for t in range(total_control_steps):
        current_u = current_state[3][0]
        u_err = desire_u - current_u

        # control every control_step seconds
        if t % control_every == 0:
            # calculate new tau
            control_signal = u_controller.step(sp=desire_u, y=current_u)

            left = control_signal
            right = control_signal
            tau = thruster.newton_to_tau(left, right)

        # record
        ts.append(t)
        us.append(current_u)
        u_errs.append(u_err)
        control_signals.append(control_signal)

        # apply tau and step
        current_state = simulator.step(tau)

    return np.array(ts)*time_step, us, u_errs, control_signals

if __name__ == "__main__":
    # single loop pid
    time_step = 0.1
    total_control_steps = 30
    control_step = 0.2
    control_every = int(control_step / time_step)

    desire_us = [0.4, 0.6, 0.8, 1.0, 1.2]

    max_base_N = 100.0
    kp = 150
    ki = 200
    kd = 5

    fig, axs = plt.subplots(3, 1)

    for desire_u in desire_us:
        ts, us, u_errs, control_sigs = u_experiment(time_step=time_step, total_control_steps=total_control_steps, control_step=control_step,
                     desire_u=desire_u, max_base_N=max_base_N, kp=kp, ki=ki, kd=kd)
    
        axs[0].plot(ts, us, label=f"$u={desire_u}$")
        axs[0].legend()
        axs[1].plot(ts, u_errs, label=f"$u={desire_u}$ error")
        axs[1].legend()
        axs[2].plot(ts, control_sigs, label=f"$u={desire_u}$ control signal")
        axs[2].legend()

    plt.show()