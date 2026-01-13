import numpy as np
from matplotlib import pyplot as plt
import plot_utils

from MarineVesselModels.simulator import NoisyVesselSimulator, SimplifiedEnvironmentalDisturbanceSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen
from MarineVesselModels.noises import GaussMarkovNoiseGenerator

from control.pid import DoubleLoopHeadingPID, PIDAW
from control.los import LOSGuider, FixedDistLOSGuider, DynamicDistLOSGuider


def test_guider(simulator, thruster, diff_controller, u_controller, guider,
                current_state, total_exp_steps, control_every):
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
    
    # Initialize control variables (will be set on first control step)
    psi_err = 0.0
    left = 0.0
    right = 0.0
    tau = np.zeros((3, 1))
    
    for t in range(total_exp_steps):
        current_x = current_state[0][0]
        current_y = current_state[1][0]
        current_psi = current_state[2][0]
        current_u = current_state[3][0]
        current_v = current_state[4][0]
        current_r = current_state[5][0]

        # control every control_step seconds
        if t % control_every == 0:
            is_ended, desired_psi = guider.step((current_x, current_y), current_psi)
            if is_ended:
                break
            # calculate new tau
            #control_signal = diff_controller.control(error=psi_err)
            diff_control_signal, ref_r = diff_controller.step(psi_ref=desired_psi, psi=current_psi, r=current_r)
            velo_control_signal = u_controller.step(sp=desire_u, y=current_u)
            # truncate
            # control_signal = max_diff_N/2 if control_signal > max_diff_N/2 else control_signal
            # control_signal = -max_diff_N/2 if control_signal < -max_diff_N/2 else control_signal

            psi_err = ((desired_psi - current_psi) + np.pi) % (2*np.pi) - np.pi

            left = velo_control_signal + diff_control_signal
            right = velo_control_signal - diff_control_signal
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
    
        # apply tau and step
        current_state = simulator.step(tau)
    return xs, ys


if __name__ == "__main__":
    time_step = 0.1
    total_exp_steps = 3000
    control_step = 0.2
    control_every = int(control_step / time_step)

    desire_u = 0.5
    max_base_N = 100.0
    max_diff_N = 60.0   # max diff between thrusters

    reached_threshold = 1.5
    forward_dist = 2.0
    
    # Environmental disturbance parameters
    env_force_magnitude = 5.0  # Newtons
    env_force_direction = np.pi/4  # radians (45°, northeast)

    # Setup noise generator
    tau_vec = np.array([60.0, 60.0, 30.0])  # s for X,Y,N
    sigmas = np.array([5.0, 5.0, 1.0])     # N, N, N·m (steady-obsv std)
    Sigma = np.diag(sigmas**2)
    tau_noise_gen = GaussMarkovNoiseGenerator(dt=time_step, Sigma=Sigma, tau=tau_vec)

    # Waypoints for all tests
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
    
    way_xs = [pt[0] for pt in waypoints]
    way_ys = [pt[1] for pt in waypoints]

    # Initialize figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    
    # Set axis limits with margin
    margin = 5.0
    x_min = min(0, min(way_xs)) - margin
    x_max = max(0, max(way_xs)) + margin
    y_min = min(0, min(way_ys)) - margin
    y_max = max(0, max(way_ys)) + margin
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(x_min, x_max)
    
    # Add background arrows showing environmental force direction
    plot_utils.add_force_direction_arrows(
        ax, direction_angle=env_force_direction,
        arrow_length=3.0, grid_spacing=8.0,
        arrow_color='lightgray', alpha=0.3, linewidth=1.0,
        coord_system='NED'
    )
    
    # Add a single arrow for legend
    legend_x = y_min + 2.0
    legend_y = x_min + 2.0
    plot_utils.add_single_force_arrow(
        ax, legend_x, legend_y, direction_angle=env_force_direction,
        arrow_length=4.0, arrow_color='lightgray', alpha=0.7,
        linewidth=2.5, label=f'Environmental force ({env_force_direction/np.pi*180:.0f}°)',
        coord_system='NED'
    )
    
    ax.scatter(way_ys, way_xs, marker="x", label="Desired waypoints", color="black", s=100)
    ax.plot(way_ys, way_xs, "--", color="black", alpha=0.5)

    # Original colors for undisturbed trajectories (blue shades)
    undisturbed_colors = ["#9EC9E6", "#3E92CC", "#1C4A69"]
    disturbed_colors = plt.get_cmap('Reds')(np.linspace(0.4, 0.8, 3))
    
    # Test each guider type
    guider_types = [
        ("Naive LOS", LOSGuider, {"output_err_flag": False}),
        ("Fixed LOS", FixedDistLOSGuider, {"output_err_flag": False, "los_dist": 2.5}),
        ("Dynamic LOS", DynamicDistLOSGuider, {"output_err_flag": False, "forward_dist": forward_dist}),
    ]

    # Run tests with undisturbed simulator (NoisyVesselSimulator)
    print("Running undisturbed tests...", flush=True)
    for idx, (guider_name, GuiderClass, guider_kwargs) in enumerate(guider_types):
        current_state = np.array([0, 0, -np.pi, 0, 0, 0]).reshape([6, 1])
        
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
        
        guider = GuiderClass(waypoints=waypoints, reached_threshold=reached_threshold, **guider_kwargs)
        xs, ys = test_guider(simulator, thruster, diff_controller, u_controller, guider,
                    current_state, total_exp_steps, control_every)
        
        ax.plot(ys, xs, label=f"{guider_name} (no disturbance)", color=undisturbed_colors[idx], linewidth=2, alpha=0.9)

    # Run tests with environmental disturbance simulator
    print("Running environmental disturbance tests...", flush=True)
    for idx, (guider_name, GuiderClass, guider_kwargs) in enumerate(guider_types):
        current_state = np.array([0, 0, -np.pi, 0, 0, 0]).reshape([6, 1])
        
        simulator = SimplifiedEnvironmentalDisturbanceSimulator(
            hydro_params=sample_hydro_params_2,
            time_step=time_step,
            env_force_magnitude=env_force_magnitude,
            env_force_direction=env_force_direction,
            model=Fossen,
            init_state=current_state,
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
        
        guider = GuiderClass(waypoints=waypoints, reached_threshold=reached_threshold, **guider_kwargs)
        xs, ys = test_guider(simulator, thruster, diff_controller, u_controller, guider,
                    current_state, total_exp_steps, control_every)
        
        ax.plot(ys, xs, label=f"{guider_name} (env disturbance)", color=disturbed_colors[idx], linewidth=2, alpha=0.9)

    ax.set_xlabel("Y position (m)")
    ax.set_ylabel("X position (m)")
    ax.set_title(f"LOS Guidance with Environmental Disturbance\n" +
                 f"Env force: {env_force_magnitude} N at {env_force_direction/np.pi*180:.0f}°, " +
                 f"Control noise: Gauss-Markov")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()