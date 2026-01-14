import numpy as np
from matplotlib import pyplot as plt
import plot_utils

from MarineVesselModels.simulator import NoisyVesselSimulator, SimplifiedEnvironmentalDisturbanceSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen
from MarineVesselModels.noises import GaussMarkovNoiseGenerator

from control.pid import DoubleLoopHeadingPID, PIDAW
from control.los import LOSGuider, FixedDistLOSGuider, DynamicDistLOSGuider, AdaptiveLOSGuider


def compute_cross_track_error(guider):
    """
    Compute signed cross-track error y_e in path-tangential frame.

    :param guider: LOSGuider instance with former_waypoint, current_waypoint, cur_pos attributes
    :return: Cross-track error in meters (positive to port side)
    """
    if guider.former_waypoint is None or guider.current_waypoint is None or guider.cur_pos is None:
        return 0.0
    line_pt1 = guider.former_waypoint
    line_pt2 = guider.current_waypoint
    pos = guider.cur_pos

    delta_x = line_pt2[0] - line_pt1[0]
    delta_y = line_pt2[1] - line_pt1[1]
    pi_h = np.arctan2(delta_y, delta_x)

    dx = pos[0] - line_pt1[0]
    dy = pos[1] - line_pt1[1]
    cross_track_err = -np.sin(pi_h)*dx + np.cos(pi_h)*dy
    return cross_track_err


def test_guider(simulator, thruster, diff_controller, u_controller, guider,
                current_state, total_exp_steps, control_every):
    """
    Run guidance simulation and collect results.

    :param simulator: Vessel simulator
    :param thruster: Thruster model
    :param diff_controller: Heading controller
    :param u_controller: Velocity controller
    :param guider: LOS guider instance
    :param current_state: Initial state vector
    :param total_exp_steps: Maximum simulation steps
    :param control_every: Control period in steps
    :return: Dictionary with simulation results
    """
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
    y_es = []  # cross-track errors
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
            diff_control_signal, ref_r = diff_controller.step(psi_ref=desired_psi, psi=current_psi, r=current_r)
            velo_control_signal = u_controller.step(sp=desire_u, y=current_u)

            psi_err = ((desired_psi - current_psi) + np.pi) % (2*np.pi) - np.pi

            left = velo_control_signal + diff_control_signal
            right = velo_control_signal - diff_control_signal
            tau = thruster.newton_to_tau(left, right)

        # record cross-track error (at each step)
        y_es.append(compute_cross_track_error(guider))

        # record other states
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

    return {
        'xs': xs, 'ys': ys, 'steps': len(xs),
        'y_es': y_es, 'ts': ts,
        'psis': psis, 'psi_errs': psi_errs,
        'us': us, 'vs': vs, 'rs': rs
    }


if __name__ == "__main__":
    time_step = 0.1
    total_exp_steps = 10000
    control_step = 0.2
    control_every = int(control_step / time_step)

    desire_u = 0.5
    max_base_N = 100.0
    max_diff_N = 60.0   # max diff between thrusters

    reached_threshold = 5.0  # Larger threshold for long distance
    forward_dist = 5.0
    nominal_los_dist = 5.0  # For Fixed LOS

    # Environmental disturbance parameters
    env_force_magnitude = 16.0  # Newtons
    env_force_direction = np.pi/4*3  # radians (135°)

    # Setup noise generator (optional)
    tau_vec = np.array([60.0, 60.0, 30.0])  # s for X,Y,N
    sigmas = np.array([5.0, 5.0, 1.0])     # N, N, N·m (steady-obsv std)
    Sigma = np.diag(sigmas**2)
    tau_noise_gen = GaussMarkovNoiseGenerator(dt=time_step, Sigma=Sigma, tau=tau_vec)

    # Define two scenarios
    scenarios = [
        {
            'name': 'Straight line',
            'waypoints': [(0, 0), (100, 100)],
        },
        {
            'name': 'Zigzag path',
            'waypoints': [(60, 15), (0, 30), (45, 45), (0, 60), (30, 75), (0, 90), (15, 105), (0, 120)],
        }
    ]

    # Guider configurations
    guider_types = [
        ("Naive LOS", LOSGuider, {"output_err_flag": False}),
        ("Fixed LOS", FixedDistLOSGuider, {"output_err_flag": False, "los_dist": nominal_los_dist}),
        ("Dynamic LOS", DynamicDistLOSGuider, {"output_err_flag": False, "forward_dist": forward_dist}),
        ("Adaptive LOS", AdaptiveLOSGuider, {"output_err_flag": False, "forward_dist": forward_dist,
                                             "gamma": 0.005, "beta_hat0": 0.0, "dt": control_step}),
    ]

    # Colors for different guiders
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(guider_types)))

    # Initialize 2x2 plot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_path1, ax_path2, ax_err1, ax_err2 = axes.flatten()

    # Store results for error plots
    all_results = {scenario['name']: {} for scenario in scenarios}

    # Run simulations for each scenario
    for scenario_idx, scenario in enumerate(scenarios):
        waypoints = scenario['waypoints']
        scenario_name = scenario['name']

        # Determine which axes to use
        if scenario_idx == 0:
            ax_path = ax_path1
            ax_err = ax_err1
        else:
            ax_path = ax_path2
            ax_err = ax_err2

        # Plot waypoints
        way_xs = [pt[0] for pt in waypoints]
        way_ys = [pt[1] for pt in waypoints]

        # Set axis limits with margin
        margin = 20.0
        x_min = min(0, min(way_xs)) - margin
        x_max = max(0, max(way_xs)) + margin
        y_min = min(0, min(way_ys)) - margin
        y_max = max(0, max(way_ys)) + margin
        ax_path.set_xlim(y_min, y_max)
        ax_path.set_ylim(x_min, x_max)
        ax_path.set_aspect("equal")

        # Add background arrows showing environmental force direction
        plot_utils.add_force_direction_arrows(
            ax=ax_path,
            direction_angle=env_force_direction,
            spacing=10.0,
        )

        ax_path.scatter(way_ys, way_xs, marker="x", label="Waypoints", color="black", s=100)
        ax_path.plot(way_ys, way_xs, "--", color="black", alpha=0.5)

        print(f"\nRunning {scenario_name} scenario...", flush=True)

        # Test each guider type
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
                        psi_kp=1, psi_ki=0.2, psi_kd=0.05,
                        r_ref_lim=0.45,
                        r_kp=200, r_ki=250, r_kd=0,
                        u_lim=max_diff_N/2,
                        r_ref_slew=None,
                        u_slew=None,
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
            result = test_guider(simulator, thruster, diff_controller, u_controller, guider,
                        current_state, total_exp_steps, control_every)

            # Store results for error plotting
            all_results[scenario_name][guider_name] = result

            # Plot path
            ax_path.plot(result['ys'], result['xs'], label=guider_name,
                        color=colors[idx], linewidth=2, alpha=0.9)

            # Plot cross-track error over time (control steps)
            t_control = np.arange(0, len(result['y_es'])) * time_step
            ax_err.plot(t_control, result['y_es'], label=guider_name,
                       color=colors[idx], linewidth=1.5, alpha=0.8)

            print(f"  {guider_name}: {result['steps']} steps", flush=True)

        # Configure path plot
        ax_path.set_xlabel("Y position (m)")
        ax_path.set_ylabel("X position (m)")
        ax_path.set_title(f"{scenario_name}\nEnv force: {env_force_magnitude} N at {env_force_direction/np.pi*180:.0f}°")
        ax_path.legend(loc='upper left')
        ax_path.grid(True, alpha=0.3)

        # Configure error plot
        ax_err.set_xlabel("Time (s)")
        ax_err.set_ylabel("Cross-track error y_e (m)")
        ax_err.set_title(f"{scenario_name} - Cross-track Error")
        ax_err.legend(loc='upper right')
        ax_err.grid(True, alpha=0.3)
        ax_err.set_ylim(-10, 10)  # Consistent y-limits for comparison

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
