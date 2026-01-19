import numpy as np
from matplotlib import pyplot as plt
import plot_utils

from MarineVesselModels.simulator import NoisyVesselSimulator, SimplifiedEnvironmentalDisturbanceSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen
from MarineVesselModels.noises import GaussMarkovNoiseGenerator

from control.pid import DoubleLoopHeadingPID, PIDAW
from control.los import AdaptiveLOSGuider, EnhancedAdaptiveLOSGuider


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


def test_guider_with_beta(simulator, thruster, diff_controller, u_controller, guider,
                          current_state, total_exp_steps, control_every, time_step,
                          pose_draw_interval=None):
    """
    Run guidance simulation and collect results including beta_hat history.

    :param simulator: Vessel simulator
    :param thruster: Thruster model
    :param diff_controller: Heading controller
    :param u_controller: Velocity controller
    :param guider: LOS guider instance
    :param current_state: Initial state vector
    :param total_exp_steps: Maximum simulation steps
    :param control_every: Control period in steps
    :return: Dictionary with simulation results including beta_hats if available
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
    vel_dirs = []  # velocity direction χ = atan2(v, u) in NED coordinates
    y_es = []  # cross-track errors
    beta_hats = []  # sideslip estimate history
    is_ended = False

    # Pose drawing setup
    pose_draw_indices = []
    if pose_draw_interval is not None and pose_draw_interval > 0:
        pose_draw_every_steps = int(pose_draw_interval / time_step)
    else:
        pose_draw_every_steps = None

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
        
        # Record beta_hat if guider has the attribute
        if hasattr(guider, 'beta_hat'):
            beta_hats.append(guider.beta_hat)
        else:
            beta_hats.append(0.0)

        # Calculate velocity direction χ = ψ + β in NED coordinates, where β = atan2(v, u) is sideslip
        if abs(current_u) < 1e-6 and abs(current_v) < 1e-6:
            vel_dir = current_psi  # If no velocity, use heading
        else:
            # Sideslip angle β = atan2(v, u) in body-fixed frame
            beta = np.arctan2(current_v, current_u)
            # Normalize heading to [-π, π] before adding
            psi_norm = ((current_psi + np.pi) % (2*np.pi)) - np.pi
            # Velocity direction in NED: χ = ψ + β
            vel_dir = psi_norm + beta
            # Normalize to [-π, π]
            vel_dir = ((vel_dir + np.pi) % (2*np.pi)) - np.pi
        vel_dirs.append(vel_dir)

        # Record pose drawing indices at specified interval
        if pose_draw_every_steps is not None and t % pose_draw_every_steps == 0:
            pose_draw_indices.append(t)

        # apply tau and step
        current_state = simulator.step(tau)

    return {
        'xs': xs, 'ys': ys, 'steps': len(xs),
        'y_es': y_es, 'ts': ts,
        'psis': psis, 'psi_errs': psi_errs,
        'us': us, 'vs': vs, 'rs': rs,
        'vel_dirs': vel_dirs,
        'beta_hats': beta_hats,
        'pose_draw_indices': pose_draw_indices
    }


if __name__ == "__main__":
    import os
    TEST_MODE = os.environ.get('TEST_MODE', '0') == '1'
    
    time_step = 0.1
    total_exp_steps = 100 if TEST_MODE else 10000
    control_step = 0.2
    control_every = int(control_step / time_step)

    desire_u = 0.5
    max_base_N = 100.0
    max_diff_N = 60.0   # max diff between thrusters

    reached_threshold = 5.0  # Larger threshold for long distance
    forward_dist = 5.0

    # Ship pose drawing parameters (for ALOS visualization)
    pose_draw_interval = 5.0  # seconds between pose drawings
    ship_radius = 0.5  # meters (ship length scale)
    heading_line_length = 0.75  # meters
    velocity_line_length = 0.75  # meters

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

    # Guider configurations - compare original ALOS vs Enhanced ALOS
    guider_types = [
        ("Adaptive LOS", AdaptiveLOSGuider, {
            "output_err_flag": False,
            "forward_dist": forward_dist,
            "gamma": 0.005,
            "beta_hat0": 0.0,
            "dt": control_step,
            "reset_beta_on_segment_change": False
        }),
        ("Enhanced ALOS", EnhancedAdaptiveLOSGuider, {
            "output_err_flag": False,
            "forward_dist": forward_dist,
            "gamma": 0.005,
            "beta_hat0": 0.0,
            "dt": control_step,
            "beta_max": np.deg2rad(30.0),
            "psi_err_threshold": np.deg2rad(15.0),
            "alpha": 0.5
        }),
    ]

    # Colors for different guiders
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(guider_types)))

    # Store results for error plots
    all_results = {scenario['name']: {} for scenario in scenarios}

    # Run simulations for each scenario
    for scenario_idx, scenario in enumerate(scenarios):
        waypoints = scenario['waypoints']
        scenario_name = scenario['name']

        # Create separate figures: one for trajectory, one for error/beta metrics
        fig_traj, ax_path = plt.subplots(1, 1, figsize=(10, 8))
        fig_metrics, axes_metrics = plt.subplots(2, 1, figsize=(10, 8))
        ax_err, ax_beta = axes_metrics.flatten()

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
            # Draw poses for both guiders
            current_pose_draw_interval = pose_draw_interval
            result = test_guider_with_beta(simulator, thruster, diff_controller, u_controller, guider,
                        current_state, total_exp_steps, control_every, time_step,
                        pose_draw_interval=current_pose_draw_interval)

            # Store results for plotting
            all_results[scenario_name][guider_name] = result

            # Plot path
            ax_path.plot(result['ys'], result['xs'], label=guider_name,
                        color=colors[idx], linewidth=2, alpha=0.9)

            # Draw ship poses (velocity line disabled)
            if 'pose_draw_indices' in result:
                # Create modified color for pose (darker version of trajectory color)
                traj_color = colors[idx]  # RGBA array
                # Darken RGB components, keep alpha
                pose_color = traj_color * np.array([0.6, 0.6, 0.6, 1.0])
                for idx_draw in result['pose_draw_indices']:
                    i = idx_draw  # index in arrays
                    pos = (result['xs'][i], result['ys'][i])  # NED coordinates (x_north, y_east)
                    heading_deg = result['psis'][i]
                    # Convert to radians and normalize to [-π, π]
                    heading_ned = ((heading_deg * np.pi / 180 + np.pi) % (2*np.pi)) - np.pi
                    vel_dir_ned = result['vel_dirs'][i]
                    
                    # Draw ship pose with specified dimensions (velocity line disabled)
                    plot_utils.draw_ship_pose_ned(
                        ax=ax_path,
                        pos=pos,
                        heading_ned=heading_ned,
                        vel_dir_ned=vel_dir_ned,
                        radius=ship_radius,
                        head_len=heading_line_length,
                        vel_len=velocity_line_length,
                        head_color=pose_color,
                        vel_color="r",
                        lw=1.5,
                        zorder=20,
                        draw_vel_line=False
                    )

            # Plot cross-track error over time (control steps)
            t_control = np.arange(0, len(result['y_es'])) * time_step
            ax_err.plot(t_control, result['y_es'], label=guider_name,
                       color=colors[idx], linewidth=1.5, alpha=0.8)
            
            # Plot beta_hat adaptation over time
            if 'beta_hats' in result:
                ax_beta.plot(t_control, np.rad2deg(result['beta_hats']), label=guider_name,
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
        
        # Configure beta_hat plot
        ax_beta.set_xlabel("Time (s)")
        ax_beta.set_ylabel("Sideslip estimate β̂ (degrees)")
        ax_beta.set_title(f"{scenario_name} - Sideslip Adaptation")
        ax_beta.legend(loc='upper right')
        ax_beta.grid(True, alpha=0.3)
        # Add horizontal lines for beta_max limits
        beta_max_deg = 30.0
        ax_beta.axhline(y=beta_max_deg, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        ax_beta.axhline(y=-beta_max_deg, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        ax_beta.set_ylim(-beta_max_deg * 1.2, beta_max_deg * 1.2)

        # Adjust layout for both figures
        fig_traj.tight_layout()
        fig_metrics.tight_layout()

        # Save figures if using non-interactive backend
        import matplotlib
        if matplotlib.get_backend().lower() == 'agg':
            fig_traj.savefig(f'enhanced_alos_trajectory_{scenario_name}.png', dpi=150)
            fig_metrics.savefig(f'enhanced_alos_metrics_{scenario_name}.png', dpi=150)
            plt.close(fig_traj)
            plt.close(fig_metrics)

    # Print performance metrics
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    for scenario_name, guider_results in all_results.items():
        print(f"\nScenario: {scenario_name}")
        print("-"*40)
        for guider_name, result in guider_results.items():
            y_es = np.array(result['y_es'])
            mean_abs_err = np.mean(np.abs(y_es))
            max_abs_err = np.max(np.abs(y_es))
            print(f"  {guider_name}:")
            print(f"    Mean |cross-track error|: {mean_abs_err:.3f} m")
            print(f"    Max  |cross-track error|: {max_abs_err:.3f} m")
            if 'beta_hats' in result:
                beta_hats = np.array(result['beta_hats'])
                mean_beta = np.mean(beta_hats)
                std_beta = np.std(beta_hats)
                max_beta = np.max(np.abs(beta_hats))
                print(f"    Mean β̂: {np.rad2deg(mean_beta):.2f}°")
                print(f"    Std β̂: {np.rad2deg(std_beta):.2f}°")
                print(f"    Max |β̂|: {np.rad2deg(max_beta):.2f}°")
                # Count how many times beta_hat hit limit (±30°)
                limit = np.deg2rad(30.0)
                hit_limit = np.sum(np.abs(beta_hats) >= limit * 0.99)
                total = len(beta_hats)
                print(f"    β̂ hit limit: {hit_limit}/{total} steps ({hit_limit/total*100:.1f}%)")

    # Display all figures
    import matplotlib
    if matplotlib.get_backend().lower() != 'agg':
        plt.show()