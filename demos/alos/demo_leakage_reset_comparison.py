#!/usr/bin/env python3
"""
Compare reset mechanisms and leakage in Enhanced Adaptive LOS across multiple current conditions.

Tests:
- Reset mechanisms without leakage: hard reset (alpha=0.0), soft reset (alpha=0.5)
- Leakage with no reset: no reset (alpha=1.0) with sigma = 0.0, 0.1, 0.01, 0.005, 0.003, 0.001
- Single scenario: Zigzag path with varying ocean current directions
- Current conditions: 7 directions at 0.1, 0.2, 0.3, 0.4 m/s (28 total conditions)
- Parallel execution: Uses multiprocessing with up to 12 processes for faster simulation

Outputs:
- Average RMSE across all current conditions for each configuration
- Performance statistics (std dev, min, max)
- Summary table of aggregated performance
- Representative plot for 0.4 m/s at 45° current condition
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from matplotlib import pyplot as plt
import plot_utils
import multiprocessing as mp
from functools import partial

from MarineVesselModels.simulator import NoisyVesselSimulator, SimplifiedEnvironmentalDisturbanceSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen, FossenWithCurrent
from MarineVesselModels.noises import GaussMarkovNoiseGenerator

from control.pid import DoubleLoopHeadingPID, PIDAW
from control.los import EnhancedAdaptiveLOSGuider


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
                          desire_u, pose_draw_interval=None):
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


def compute_rmse(y_es):
    """Compute Root Mean Square Error of cross-track errors."""
    if not y_es:
        return 0.0
    return np.sqrt(np.mean(np.array(y_es)**2))


def compute_mae(y_es):
    """Compute Mean Absolute Error of cross-track errors."""
    if not y_es:
        return 0.0
    return np.mean(np.abs(np.array(y_es)))


def compute_mean_error(y_es):
    """
    Compute mean cross-track error (signed average).

    Positive values indicate average bias to port side,
    negative values indicate average bias to starboard side.
    """
    if not y_es:
        return 0.0
    return np.mean(np.array(y_es))


def create_guider_configs():
    """
    Create configurations for testing.

    Returns list of (name, config_dict) tuples.

    Configurations:
    - Hard reset (α=0.0) without leakage (σ=0.0)
    - Soft reset (α=0.5) without leakage (σ=0.0)
    - No reset (α=1.0) with various leakage levels (σ=0.0, 0.1, 0.01, 0.005, 0.003, 0.001)
    """
    configs = []

    # Fixed parameters for all tests
    base_params = {
        'output_err_flag': False,
        'forward_dist': 5.0,
        'gamma': 0.005,
        'beta_hat0': 0.0,
        'dt': 0.2,  # control_step
        'beta_max': np.deg2rad(30.0),
        'psi_err_threshold': np.deg2rad(15.0),
    }

    # Hard reset without leakage
    config = base_params.copy()
    config.update({'alpha': 0.0, 'sigma': 0.0})
    configs.append(("α=0.0 (hard), σ=0.000", config))

    # Soft reset without leakage
    config = base_params.copy()
    config.update({'alpha': 0.5, 'sigma': 0.0})
    configs.append(("α=0.5 (soft), σ=0.000", config))

    # No reset with various leakage levels
    sigmas = [0.0, 0.1, 0.01, 0.005, 0.003, 0.001]
    for sigma in sigmas:
        name = f"α=1.0 (no reset), σ={sigma:.3f}"
        config = base_params.copy()
        config.update({'alpha': 1.0, 'sigma': sigma})
        configs.append((name, config))

    return configs


def run_scenario(scenario_name, waypoints, current_velocity, current_direction, configs,
                 time_step, total_exp_steps, control_step, desire_u):
    """
    Run a single scenario with all guider configurations.

    Returns dictionary mapping config_name to results.
    """
    results = {}

    # Common simulation setup for this scenario
    hydro_params = sample_hydro_params_2
    model = FossenWithCurrent
    init_state = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])

    # Current direction passed as parameter

    for config_name, config in configs:
        print(f"  Running {config_name}...")

        # Create simulator with ocean current (no environmental force)
        simulator = SimplifiedEnvironmentalDisturbanceSimulator(
            hydro_params=hydro_params,
            time_step=time_step,
            model=model,
            env_force_magnitude=0,
            env_force_direction=0,
            current_velocity=current_velocity,
            current_direction=current_direction,
        )
        simulator.state = init_state.copy()

        # Thruster model
        thruster = NaiveDoubleThruster(b=sample_b_2)

        # Controller limits (same as demo_alos_tuning)
        max_base_N = sample_thrust_2  # 100.0
        max_diff_N = 60.0

        # Controllers
        diff_controller = DoubleLoopHeadingPID(
            dt=control_step,
            psi_kp=1, psi_ki=0.2, psi_kd=0.05,
            r_ref_lim=0.45,
            r_kp=200, r_ki=250, r_kd=0,
            u_lim=max_diff_N/2,
            r_ref_slew=None,
            u_slew=None,
        )
        u_controller = PIDAW(kp=150, ki=200, kd=5, dt=control_step, u_min=-max_base_N, u_max=max_base_N)

        # Create guider
        guider = EnhancedAdaptiveLOSGuider(
            waypoints=waypoints,
            reached_threshold=5.0,
            **config
        )

        # Run simulation
        result = test_guider_with_beta(
            simulator=simulator,
            thruster=thruster,
            diff_controller=diff_controller,
            u_controller=u_controller,
            guider=guider,
            current_state=init_state.copy(),
            total_exp_steps=total_exp_steps,
            control_every=int(control_step / time_step),
            time_step=time_step,
            desire_u=desire_u,
        )

        results[config_name] = result

    return results


def plot_results(scenario_name, waypoints, results, output_prefix, current_direction, time_step):
    """
    Create plots for a scenario.

    Creates:
    1. Trajectory plot with waypoints and current direction arrows
    2. Cross-track error over time
    3. Sideslip estimate (beta_hat) over time

    :param time_step: Simulation time step (seconds)
    """
    n_configs = len(results)

    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{scenario_name} - Reset & Leakage Comparison', fontsize=16)

    # Colors for different configurations
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, n_configs))

    # Plot 1: Trajectories (NED coordinates: X North, Y East)
    ax = axs[0, 0]
    # Plot waypoints (swap axes for plotting: Y East vs X North)
    wp_x = [pt[0] for pt in waypoints]  # X = North
    wp_y = [pt[1] for pt in waypoints]  # Y = East
    ax.plot(wp_y, wp_x, 'k--', alpha=0.5, label='Waypoints')
    ax.scatter(wp_y, wp_x, c='red', s=50, zorder=5)

    # Plot trajectories for each configuration (swap axes)
    for idx, (config_name, result) in enumerate(results.items()):
        xs = result['xs']  # X positions (North)
        ys = result['ys']  # Y positions (East)
        ax.plot(ys, xs, color=colors[idx], alpha=0.7, linewidth=1.5, label=config_name)

    # Add current direction arrows (background)
    plot_utils.add_force_direction_arrows(
        ax=ax,
        direction_angle=current_direction,
        spacing=10.0,
    )

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('Vessel Trajectories (NED coordinates)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Plot 2: Cross-track error over time
    ax = axs[0, 1]
    for idx, (config_name, result) in enumerate(results.items()):
        ts = np.array(result['ts']) * time_step
        y_es = result['y_es']
        ax.plot(ts, y_es, color=colors[idx], alpha=0.7, linewidth=1.5, label=config_name)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cross-track error (m)')
    ax.set_title('Cross-track Error Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    # Plot 3: Sideslip estimate (beta_hat) over time
    ax = axs[1, 0]
    for idx, (config_name, result) in enumerate(results.items()):
        ts = np.array(result['ts']) * time_step
        beta_hats = np.array(result['beta_hats'])
        ax.plot(ts, beta_hats, color=colors[idx], alpha=0.7, linewidth=1.5, label=config_name)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('β̂ (rad)')
    ax.set_title('Sideslip Estimate (β̂) Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    # Plot 4: RMSE comparison (bar chart)
    ax = axs[1, 1]
    config_names = list(results.keys())
    rmse_values = [compute_rmse(results[name]['y_es']) for name in config_names]

    bars = ax.bar(range(len(config_names)), rmse_values, color=colors[:len(config_names)])
    ax.set_xlabel('Configuration')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('Cross-track Error RMSE Comparison')
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Add RMSE values on top of bars
    for bar, rmse in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rmse:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Show or close figure based on backend (like ablation study)
    import matplotlib
    if matplotlib.get_backend().lower() == 'agg':
        # Non-interactive backend: close figure without saving
        plt.close()
        print(f"  Note: Non-interactive backend '{matplotlib.get_backend()}'. Figure not displayed.")
    else:
        # Interactive backend: show figure
        plt.show()
        plt.close()





def compute_average_metrics(condition_results):
    """
    Compute average metrics across multiple current conditions.

    Args:
        condition_results: List of dictionaries, each containing:
            - 'velocity': current velocity magnitude (m/s)
            - 'angle': current direction (radians)
            - 'results': dictionary mapping config_name to result dict

    Returns:
        Dictionary mapping config_name to aggregated metrics
    """
    aggregated = {}

    # First, collect all metric values for each configuration across all conditions
    for condition in condition_results:
        for config_name, result in condition['results'].items():
            if config_name not in aggregated:
                aggregated[config_name] = {
                    'rmse_values': [],
                    'mae_values': [],
                    'mean_err_values': [],
                    'velocities': [],
                    'angles': [],
                    'condition_results': []  # store individual condition results
                }

            rmse = compute_rmse(result['y_es'])
            mae = compute_mae(result['y_es'])
            mean_err = compute_mean_error(result['y_es'])

            aggregated[config_name]['rmse_values'].append(rmse)
            aggregated[config_name]['mae_values'].append(mae)
            aggregated[config_name]['mean_err_values'].append(mean_err)
            aggregated[config_name]['velocities'].append(condition['velocity'])
            aggregated[config_name]['angles'].append(condition['angle'])
            aggregated[config_name]['condition_results'].append({
                'velocity': condition['velocity'],
                'angle': condition['angle'],
                'rmse': rmse,
                'mae': mae,
                'mean_err': mean_err,
                'result': result
            })

    # Compute statistics for each configuration
    for config_name, data in aggregated.items():
        rmse_values = np.array(data['rmse_values'])
        mae_values = np.array(data['mae_values'])
        mean_err_values = np.array(data['mean_err_values'])

        data['avg_rmse'] = np.mean(rmse_values)
        data['std_rmse'] = np.std(rmse_values)
        data['min_rmse'] = np.min(rmse_values)
        data['max_rmse'] = np.max(rmse_values)

        data['avg_mae'] = np.mean(mae_values)
        data['std_mae'] = np.std(mae_values)
        data['min_mae'] = np.min(mae_values)
        data['max_mae'] = np.max(mae_values)

        data['avg_mean_err'] = np.mean(mean_err_values)
        data['std_mean_err'] = np.std(mean_err_values)
        data['min_mean_err'] = np.min(mean_err_values)
        data['max_mean_err'] = np.max(mean_err_values)

        data['num_conditions'] = len(rmse_values)

    return aggregated














def print_averaged_metrics_table(aggregated_metrics):
    """Print table of averaged metrics across all current conditions."""
    print("\n" + "="*120)
    print("AVERAGED PERFORMANCE METRICS ACROSS ALL CURRENT CONDITIONS")
    print("="*120)

    # Find baseline (hard reset with σ=0.000)
    baseline_key = None
    for config_name in aggregated_metrics.keys():
        if "α=0.0 (hard), σ=0.000" in config_name:
            baseline_key = config_name
            break

    baseline_rmse = aggregated_metrics[baseline_key]['avg_rmse'] if baseline_key else 0.0

    # Header
    print(f"{'Configuration':<25} | {'Avg RMSE':>9} | {'Std RMSE':>9} | {'% vs base':>9} | "
          f"{'Avg MAE':>9} | {'Std MAE':>9} | {'Avg Mean Err':>12} | {'Std Mean Err':>12} | {'# Cond':>6}")
    print("-"*120)

    # Sort configurations by average RMSE (best to worst)
    configs_sorted = sorted(
        aggregated_metrics.items(),
        key=lambda x: x[1]['avg_rmse']
    )

    for config_name, metrics in configs_sorted:
        # Calculate percentage difference from baseline
        if baseline_key and baseline_rmse > 0:
            pct_diff = 100.0 * (metrics['avg_rmse'] - baseline_rmse) / baseline_rmse
            pct_str = f"{pct_diff:>+8.2f}%"
        else:
            pct_str = "    N/A  "

        print(f"{config_name:<25} | {metrics['avg_rmse']:>9.4f} | {metrics['std_rmse']:>9.4f} | {pct_str:>9} | "
              f"{metrics['avg_mae']:>9.4f} | {metrics['std_mae']:>9.4f} | "
              f"{metrics['avg_mean_err']:>12.4f} | {metrics['std_mean_err']:>12.4f} | {metrics['num_conditions']:>6}")

    print("="*120)
    if baseline_key:
        print(f"Baseline: {baseline_key} (Avg RMSE = {baseline_rmse:.4f})")
        print("Note: Mean Error positive = average bias to port side, negative = average bias to starboard side")


def _run_single_condition(args):
    """
    Helper function to run a single condition for multiprocessing.
    
    Args:
        args: Tuple containing (index, velocity, angle_deg, waypoints, configs, 
                               time_step, total_exp_steps, control_step, desire_u)
    
    Returns:
        Dictionary with condition results
    """
    i, velocity, angle_deg, waypoints, configs, time_step, total_exp_steps, control_step, desire_u = args
    angle_rad = np.deg2rad(angle_deg)
    print(f"  Condition {i+1}: {velocity} m/s at {angle_deg}°")
    
    results = run_scenario(
        scenario_name=f"Zigzag ({velocity} m/s, {angle_deg}°)",
        waypoints=waypoints,
        current_velocity=velocity,
        current_direction=angle_rad,
        configs=configs,
        time_step=time_step,
        total_exp_steps=total_exp_steps,
        control_step=control_step,
        desire_u=desire_u,
    )
    
    return {
        'velocity': velocity,
        'angle': angle_rad,
        'angle_deg': angle_deg,
        'results': results
    }


def run_multiple_conditions(waypoints, current_conditions, configs,
                            time_step, total_exp_steps, control_step, desire_u,
                            max_procs=12):
    """
    Run simulations across multiple current conditions using multiprocessing.

    Args:
        waypoints: List of (x, y) waypoints for the zigzag path
        current_conditions: List of (velocity_mps, angle_degrees) tuples
        configs: List of (name, config_dict) guider configurations
        time_step, total_exp_steps, control_step, desire_u: Simulation parameters
        max_procs: Maximum number of parallel processes (default: 12)

    Returns:
        List of condition results dictionaries
    """
    condition_results = []
    
    # Prepare arguments for multiprocessing
    args_list = []
    for i, (velocity, angle_deg) in enumerate(current_conditions):
        args_list.append((
            i, velocity, angle_deg, waypoints, configs,
            time_step, total_exp_steps, control_step, desire_u
        ))
    
    # Determine number of processes to use
    n_conditions = len(current_conditions)
    n_procs = min(max_procs, n_conditions, mp.cpu_count())
    
    if n_procs > 1:
        print(f"Running {n_conditions} conditions using {n_procs} parallel processes...")
        with mp.Pool(processes=n_procs) as pool:
            condition_results = list(pool.imap(_run_single_condition, args_list))
    else:
        print(f"Running {n_conditions} conditions sequentially...")
        for args in args_list:
            condition_results.append(_run_single_condition(args))
    
    # Sort by original order (though imap should preserve order)
    condition_results.sort(key=lambda x: current_conditions.index((x['velocity'], x['angle_deg'])))
    
    return condition_results


if __name__ == "__main__":
    import os

    # Test mode for quick debugging
    TEST_MODE = os.environ.get('TEST_MODE', '0') == '1'

    # Simulation parameters
    time_step = 0.1
    total_exp_steps = 100 if TEST_MODE else 15000
    control_step = 0.2
    desire_u = 0.5  # desired surge velocity (m/s)

    # Define current conditions (4 velocities × 7 angles = 28 conditions)
    current_conditions = []
    velocities = [0.1, 0.2, 0.3, 0.4]
    angles = [-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0]
    for velocity in velocities:
        for angle in angles:
            current_conditions.append((velocity, angle))

    # Zigzag path waypoints
    zigzag_waypoints = [(0, 0), (100, 100), (0, 200), (100, 300), (0, 400)]

    # Create guider configurations
    configs = create_guider_configs()
    print(f"Testing {len(configs)} configurations across {len(current_conditions)} current conditions:")
    for name, _ in configs:
        print(f"  - {name}")

    # Run simulations across all current conditions
    print(f"\nTotal simulations: {len(configs)} configs × {len(current_conditions)} conditions = {len(configs) * len(current_conditions)}")

    condition_results = run_multiple_conditions(
        waypoints=zigzag_waypoints,
        current_conditions=current_conditions,
        configs=configs,
        time_step=time_step,
        total_exp_steps=total_exp_steps,
        control_step=control_step,
        desire_u=desire_u,
        max_procs=12,
    )

    # Compute aggregated metrics
    aggregated_metrics = compute_average_metrics(condition_results)

    # Print averaged metrics table
    print_averaged_metrics_table(aggregated_metrics)

    # Optional: Generate plots for a representative condition (0.4 m/s at 45° current)
    if condition_results and not TEST_MODE:
        # Find a condition with 0.4 m/s current velocity at 45° direction
        representative_condition = None
        for condition in condition_results:
            if condition['velocity'] == 0.4 and condition['angle_deg'] == 45.0:
                representative_condition = condition
                break
        
        # If no 0.4 m/s at 45° condition found, look for any 0.4 m/s condition
        if representative_condition is None:
            for condition in condition_results:
                if condition['velocity'] == 0.4:
                    representative_condition = condition
                    break
        
        # If still no 0.4 m/s condition found, use the first one
        if representative_condition is None:
            representative_condition = condition_results[0]

        print(f"\nGenerating representative plots for condition: {representative_condition['velocity']} m/s at {representative_condition['angle_deg']}°")

        plot_results(
            scenario_name=f"Zigzag Path ({representative_condition['velocity']} m/s, {representative_condition['angle_deg']}°)",
            waypoints=zigzag_waypoints,
            results=representative_condition['results'],
            output_prefix='leakage_reset_comparison',
            current_direction=representative_condition['angle'],
            time_step=time_step
        )

    print("\nDone!")
