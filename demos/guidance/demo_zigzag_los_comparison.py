import numpy as np
from matplotlib import pyplot as plt
import plot_utils

"""
Demo: Compare zigzag path following performance under different environmental disturbance magnitudes
and across different LOS guidance methods (Naive, Fixed, Dynamic, Adaptive LOS).
Generates separate plots for each disturbance magnitude showing trajectory and cross-track error
for all LOS methods on the same axes for comparison.
"""

from MarineVesselModels.simulator import SimplifiedEnvironmentalDisturbanceSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen


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
                current_state, total_exp_steps, control_every, desire_u):
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
    :param desire_u: Desired surge velocity (m/s)
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
    import os
    TEST_MODE = os.environ.get('TEST_MODE', '0') == '1'
    # Simulation parameters
    time_step = 0.1
    total_exp_steps = 20000
    if TEST_MODE:
        total_exp_steps = 100
    control_step = 0.2
    control_every = int(control_step / time_step)

    desire_u = 0.5
    max_base_N = 100.0
    max_diff_N = 60.0   # max diff between thrusters

    reached_threshold = 5.0
    forward_dist = 5.0

    # Environmental disturbance parameters
    env_force_direction = np.pi/4*3  # radians (135°)
    disturbance_magnitudes = [0.0, 5.0, 10.0, 20.0]  # Newtons
    if TEST_MODE:
        disturbance_magnitudes = [0.0]

    # LOS guider configurations to compare
    guider_configs = [
        ("Naive LOS", LOSGuider, {"output_err_flag": False}),
        ("Fixed LOS", FixedDistLOSGuider, {"output_err_flag": False, "los_dist": forward_dist}),
        ("Dynamic LOS", DynamicDistLOSGuider, {"output_err_flag": False, "forward_dist": forward_dist}),
        ("Adaptive LOS", AdaptiveLOSGuider, {"output_err_flag": False, "forward_dist": forward_dist,
                                             "gamma": 0.005, "beta_hat0": 0.0, "dt": control_step}),
    ]

    # Zigzag waypoints (6x scale from demo_los_double_loop.py, doubled from previous 3x)
    waypoints = [
        (120, 30),
        (0, 60),
        (90, 90),
        (0, 120),
        (60, 150),
        (0, 180),
        (30, 210),
        (0, 240),
    ]

    # Store results: magnitude -> guider_name -> result
    results = {}

    print("Running zigzag path with different disturbance magnitudes and LOS methods...")

    for magnitude in disturbance_magnitudes:
        print(f"\nDisturbance magnitude: {magnitude} N")
        results[magnitude] = {}

        for guider_name, GuiderClass, guider_kwargs in guider_configs:
            print(f"  Testing {guider_name}...")

            current_state = np.array([0, 0, -np.pi, 0, 0, 0]).reshape([6, 1])

            simulator = SimplifiedEnvironmentalDisturbanceSimulator(
                hydro_params=sample_hydro_params_2,
                time_step=time_step,
                env_force_magnitude=magnitude,
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

            guider = GuiderClass(
                waypoints=waypoints,
                reached_threshold=reached_threshold,
                **guider_kwargs
            )

            result = test_guider(simulator, thruster, diff_controller, u_controller, guider,
                                 current_state, total_exp_steps, control_every, desire_u)
            results[magnitude][guider_name] = result

            print(f"    Steps completed: {result['steps']}")

    # Colors for different LOS methods
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(guider_configs)))

    # Create individual plots for each magnitude
    for magnitude, magnitude_results in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_path, ax_err = axes

        # Plot waypoints
        way_xs = [pt[0] for pt in waypoints]
        way_ys = [pt[1] for pt in waypoints]

        # Path plot setup
        ax_path.set_aspect("equal")
        ax_path.scatter(way_ys, way_xs, marker="x", label="Waypoints", color="black", s=100)
        ax_path.plot(way_ys, way_xs, "--", color="black", alpha=0.5)

        # Cross-track error plot setup
        ax_err.set_xlabel("Time (s)")
        ax_err.set_ylabel("Cross-track error y_e (m)")
        ax_err.set_title(f"Cross-track Error Comparison - Disturbance: {magnitude} N")
        ax_err.grid(True, alpha=0.3)

        # Store metrics for summary table
        metrics_data = []

        # Plot each LOS method
        for idx, (guider_name, _, _) in enumerate(guider_configs):
            if guider_name not in magnitude_results:
                continue

            result = magnitude_results[guider_name]

            # Plot trajectory
            ax_path.plot(result['ys'], result['xs'], label=guider_name,
                        color=colors[idx], linewidth=2, alpha=0.9)

            # Plot cross-track error
            t_control = np.arange(0, len(result['y_es'])) * time_step
            ax_err.plot(t_control, result['y_es'], label=guider_name,
                       color=colors[idx], linewidth=1.5, alpha=0.8)

            # Calculate metrics
            max_err = np.max(np.abs(result['y_es']))
            avg_err = np.mean(np.abs(result['y_es']))
            metrics_data.append((guider_name, max_err, avg_err, result['steps']))

        # Add force direction arrows
        plot_utils.add_force_direction_arrows(
            ax=ax_path,
            direction_angle=env_force_direction,
            spacing=20.0,
        )

        # Configure path plot
        ax_path.set_xlabel("Y position (m)")
        ax_path.set_ylabel("X position (m)")
        ax_path.set_title(f"Trajectory Comparison - Disturbance: {magnitude} N")
        ax_path.legend(loc='upper left')
        ax_path.grid(True, alpha=0.3)

        # Configure error plot
        ax_err.legend(loc='upper right')
        ax_err.set_ylim(-20, 20)  # Slightly larger limits for comparison

        # Add metrics summary as table
        if metrics_data:
            # Create metrics text
            metrics_text = "LOS Method Performance:\n"
            for guider_name, max_err, avg_err, steps in metrics_data:
                metrics_text += f"\n{guider_name}:\n"
                metrics_text += f"  Max |y_e|: {max_err:.2f} m\n"
                metrics_text += f"  Avg |y_e|: {avg_err:.2f} m\n"
                metrics_text += f"  Steps: {steps}"

            ax_err.text(0.02, 0.98, metrics_text, transform=ax_err.transAxes,
                       verticalalignment='top', horizontalalignment='left',
                       fontsize=9, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        if not TEST_MODE:
            plt.show()

    # Create summary comparison across all magnitudes
    print("\n" + "="*60)
    print("SUMMARY: LOS Method Performance Across Disturbance Magnitudes")
    print("="*60)

    for guider_name, _, _ in guider_configs:
        print(f"\n{guider_name}:")
        print("-" * 40)
        for magnitude in disturbance_magnitudes:
            if magnitude in results and guider_name in results[magnitude]:
                result = results[magnitude][guider_name]
                max_err = np.max(np.abs(result['y_es']))
                avg_err = np.mean(np.abs(result['y_es']))
                print(f"  {magnitude:4.1f} N: Max |y_e| = {max_err:6.2f} m, Avg |y_e| = {avg_err:6.2f} m, Steps = {result['steps']}")

    print("\nSimulation completed.")
