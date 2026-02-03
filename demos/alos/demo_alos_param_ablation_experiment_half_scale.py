#!/usr/bin/env python3
"""
ALOS parameter ablation experiment per ablation_instruction.md.

Experiment design:
- Parameter grid: 3 reset mechanisms × 6 leakage values = 18 combos
- Environment grid: 4 current speeds × 6 directions = 24 scenarios
- Single zigzag path
- Fixed: Clamp=ON (30°), Conditional integration=ON
- Total runs: 18 × 24 = 432

Outputs:
1. results_summary.csv with metrics for each run
2. Heatmap PNGs for each parameter configuration
3. report.md with analysis and top combinations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import multiprocessing as mp
from functools import partial
import csv
import json
from datetime import datetime
from pathlib import Path

import plot_utils

from MarineVesselModels.simulator import SimplifiedEnvironmentalDisturbanceSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, FossenWithCurrent

from control.pid import DoubleLoopHeadingPID, PIDAW
from control.los import EnhancedAdaptiveLOSGuider


def run_simulation_wrapper(config, waypoints, time_step, total_exp_steps, control_step, desire_u):
    """
    Wrapper for multiprocessing that unpacks config dict.
    """
    return run_single_simulation(
        run_id=config['run_id'],
        param_config=config['param_config'],
        env_config=config['env_config'],
        waypoints=waypoints,
        time_step=time_step,
        total_exp_steps=total_exp_steps,
        control_step=control_step,
        desire_u=desire_u
    )


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


def test_guider_with_metrics(simulator, thruster, diff_controller, u_controller, guider,
                             current_state, total_exp_steps, control_every, time_step,
                             desire_u):
    """
    Run guidance simulation and collect cross-track error metrics.

    Returns dictionary with metrics and error history.
    """
    y_es = []  # cross-track errors at each simulation step
    xs = []    # x positions (north)
    ys = []    # y positions (east)
    is_ended = False

    # Initialize control variables
    psi_err = 0.0
    left = 0.0
    right = 0.0
    tau = np.zeros((3, 1))

    # Early termination checks
    max_steps_exceeded = False
    numerical_error = False
    diverged = False

    t = 0  # Initialize step counter
    for step_idx in range(total_exp_steps):
        t = step_idx  # Update step counter
        current_x = current_state[0][0]
        current_y = current_state[1][0]
        current_psi = current_state[2][0]
        current_u = current_state[3][0]
        current_v = current_state[4][0]
        current_r = current_state[5][0]

        # Check for numerical issues
        if not np.isfinite(current_x) or not np.isfinite(current_y):
            numerical_error = True
            break

        # Check for divergence (far from path)
        if abs(current_x) > 1000 or abs(current_y) > 1000:
            diverged = True
            break

        # Control every control_step seconds
        if t % control_every == 0:
            is_ended, desired_psi = guider.step((current_x, current_y), current_psi)

            # Early termination if path completed
            if is_ended:
                break

            # Compute cross-track error
            y_e = compute_cross_track_error(guider)

            # Velocity controller
            left_right_sum_N = u_controller.step(sp=desire_u, y=current_u)

            # Heading controller
            diff_control_signal, _ = diff_controller.step(psi_ref=desired_psi, psi=current_psi, r=current_r)
            left_right_diff_N = diff_control_signal

            # Convert to left/right thrust
            left = (left_right_sum_N + left_right_diff_N) / 2.0
            right = (left_right_sum_N - left_right_diff_N) / 2.0
            tau = thruster.newton_to_tau(l_thrust_N=left, r_thrust_N=right)

        # Store trajectory and error
        xs.append(current_x)
        ys.append(current_y)
        y_e = compute_cross_track_error(guider)
        y_es.append(y_e)

        # Apply control
        current_state = simulator.step(tau)

    # Check if max steps were reached
    max_steps_exceeded = (t == total_exp_steps - 1)

    # Calculate metrics
    y_es_array = np.array(y_es)
    if len(y_es_array) == 0:
        cte_rmse = np.nan
        cte_p95 = np.nan
    else:
        cte_rmse = np.sqrt(np.mean(y_es_array**2))
        cte_p95 = np.percentile(np.abs(y_es_array), 95)

    # Determine failure status
    is_failed = numerical_error or diverged
    fail_reason = ""
    if numerical_error:
        fail_reason = "numerical_error"
    elif diverged:
        fail_reason = "diverged"
    elif max_steps_exceeded:
        fail_reason = "max_steps_exceeded"
    elif not is_ended:
        fail_reason = "path_not_completed"

    return {
        'cte_rmse': cte_rmse,
        'cte_p95': cte_p95,
        'steps': len(y_es),
        'finished': is_ended and not is_failed,
        'is_failed': is_failed,
        'fail_reason': fail_reason,
        'xs': xs,
        'ys': ys,
        'y_es': y_es
    }


def create_parameter_grid():
    """
    Create parameter grid for the experiment.

    Returns: list of dicts with parameter configurations
    """
    reset_modes = [
        {"reset_mode": "hard_reset", "alpha": 0.0, "soft_reset_coeff": 0.0},
        {"reset_mode": "soft_reset", "alpha": 0.5, "soft_reset_coeff": 0.5},
        {"reset_mode": "no_reset", "alpha": 1.0, "soft_reset_coeff": 1.0}
    ]

    leakage_values = [0.0, 0.1, 0.01, 0.005, 0.003, 0.001]

    parameter_grid = []

    for reset_config in reset_modes:
        for leakage in leakage_values:
            param = {
                'reset_mode': reset_config['reset_mode'],
                'alpha': reset_config['alpha'],
                'soft_reset_coeff': reset_config['soft_reset_coeff'],
                'leakage': leakage,
                'sigma': leakage  # sigma is the leakage parameter in EnhancedAdaptiveLOSGuider
            }
            parameter_grid.append(param)

    return parameter_grid


def create_environment_grid():
    """
    Create environment grid with current+force disturbances.

    Returns: list of dicts with environment configurations
    """
    current_speeds = [0.1, 0.2, 0.3, 0.4]  # m/s (include 0.4)
    directions_deg = [0, 30, 90, 150, 180, 270]  # degrees

    environment_grid = []

    for current_speed in current_speeds:
        for direction_deg in directions_deg:
            # Force magnitude: (current_speed / 0.1) * 5 N
            force_magnitude = (current_speed / 0.1) * 5.0 if current_speed > 0 else 0.0

            env = {
                'current_speed': current_speed,
                'direction_deg': direction_deg,
                'force_magnitude': force_magnitude,
                'direction_rad': np.deg2rad(direction_deg)
            }
            environment_grid.append(env)

    return environment_grid


def create_guider_config(param_config):
    """
    Create EnhancedAdaptiveLOSGuider configuration from parameter dict.

    Fixed parameters:
    - Clamp ON: beta_max = 30°
    - Conditional integration ON: psi_err_threshold = 15°
    """
    config = {
        'output_err_flag': False,
        'forward_dist': 5.0,
        'gamma': 0.005,
        'beta_hat0': 0.0,
        'dt': 0.2,  # control_step
        'beta_max': np.deg2rad(30.0),  # Clamp ON
        'psi_err_threshold': np.deg2rad(15.0),  # Conditional integration ON
        'alpha': param_config['alpha'],
        'sigma': param_config['sigma']
    }
    return config


def run_single_simulation(run_id, param_config, env_config, waypoints,
                          time_step=0.01, total_exp_steps=40000, 
                          control_step=20, desire_u=0.5):
    """
    Run a single simulation with given parameters and environment.

    Returns: dict with results for this run
    """
    try:
        # Create simulator with current and force
        simulator = SimplifiedEnvironmentalDisturbanceSimulator(
            hydro_params=sample_hydro_params_2,
            time_step=time_step,
            model=FossenWithCurrent,
            env_force_magnitude=env_config['force_magnitude'],
            env_force_direction=env_config['direction_rad'],
            l_eff=0.1,  # arbitrary for now
            current_velocity=env_config['current_speed'],
            current_direction=env_config['direction_rad'],
        )

        # Initial state: heading 0, velocity 0
        init_state = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])
        simulator.state = init_state.copy()

        # Thruster model
        thruster = NaiveDoubleThruster(b=sample_b_2)

        # Controller limits
        max_base_N = sample_thrust_2  # 100.0
        max_diff_N = 60.0

        # Controllers
        diff_controller = DoubleLoopHeadingPID(
            dt=control_step*time_step,  # Convert steps to seconds
            psi_kp=1, psi_ki=0.2, psi_kd=0.05,
            r_ref_lim=0.45,
            r_kp=200, r_ki=250, r_kd=0,
            u_lim=max_diff_N/2,
            r_ref_slew=None,
            u_slew=None,
        )
        u_controller = PIDAW(kp=150, ki=200, kd=5, dt=control_step*time_step,
                           u_min=-max_base_N, u_max=max_base_N)

        # Create guider with waypoints and configuration
        guider_config = create_guider_config(param_config)
        reached_threshold = 5.0  # meters
        guider = EnhancedAdaptiveLOSGuider(
            waypoints=waypoints,
            reached_threshold=reached_threshold,
            forward_dist=guider_config['forward_dist'],
            dt=guider_config['dt'],
            gamma=guider_config['gamma'],
            sigma=guider_config['sigma'],
            beta_hat0=guider_config['beta_hat0'],
            beta_max=guider_config['beta_max'],
            psi_err_threshold=guider_config['psi_err_threshold'],
            alpha=guider_config['alpha'],
            output_err_flag=guider_config['output_err_flag']
        )

        # Run simulation
        results = test_guider_with_metrics(
            simulator=simulator,
            thruster=thruster,
            diff_controller=diff_controller,
            u_controller=u_controller,
            guider=guider,
            current_state=init_state,
            total_exp_steps=total_exp_steps,
            control_every=control_step,
            time_step=time_step,
            desire_u=desire_u
        )

        # Add run metadata
        results.update({
            'run_id': run_id,
            'reset_mode': param_config['reset_mode'],
            'soft_reset_coeff': param_config['soft_reset_coeff'],
            'leakage': param_config['leakage'],
            'alpha': param_config['alpha'],
            'sigma': param_config['sigma'],
            'current_speed': env_config['current_speed'],
            'force_magnitude': env_config['force_magnitude'],
            'direction_deg': env_config['direction_deg']
        })

        return results

    except Exception as e:
        # Catch any unexpected errors
        print(f"  Error in run {run_id}: {e}")
        return {
            'run_id': run_id,
            'reset_mode': param_config['reset_mode'],
            'soft_reset_coeff': param_config['soft_reset_coeff'],
            'leakage': param_config['leakage'],
            'current_speed': env_config['current_speed'],
            'force_magnitude': env_config['force_magnitude'],
            'direction_deg': env_config['direction_deg'],
            'cte_rmse': np.nan,
            'cte_p95': np.nan,
            'steps': 0,
            'finished': False,
            'is_failed': True,
            'fail_reason': f'exception: {str(e)[:100]}',
            'xs': [],
            'ys': [],
            'y_es': []
        }


def run_experiment_parallel(output_dir, test_mode=False):
    """
    Main experiment function with parallel execution.
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ALOS PARAMETER ABLATION EXPERIMENT")
    print("="*80)

    # Fixed waypoints (zigzag only) - HALF SCALE
    waypoints = [(0, 0), (50, 50), (0, 100), (50, 150), (0, 200)]
    print(f"Path: HALF-SCALE Zigzag with {len(waypoints)} waypoints (283m total)")

    # Simulation parameters
    time_step = 0.01
    control_step = 20  # Control every 0.2 seconds
    desire_u = 0.5  # m/s (consistent with previous ablation demo; achievable ~0.66 m/s due to thrust limit)

    if test_mode:
        total_exp_steps = 150000  # ~1500 seconds at ~0.66 m/s (guarantees path completion even with disturbance)
        print("TEST MODE: Reduced simulation steps")
    else:
        total_exp_steps = 300000  # ~3000 seconds at ~0.66 m/s (guarantees path completion even with strong current)

    # Create parameter and environment grids
    parameter_grid = create_parameter_grid()
    environment_grid = create_environment_grid()

    print(f"Parameter combos: {len(parameter_grid)} (3 reset × 6 leakage)")
    print(f"Environment scenarios: {len(environment_grid)} (4 current × 6 direction)")
    print(f"Total runs: {len(parameter_grid) * len(environment_grid)}")

    # Prepare all run configurations
    run_configs = []
    run_id = 0

    for param_idx, param_config in enumerate(parameter_grid):
        for env_idx, env_config in enumerate(environment_grid):
            run_configs.append({
                'run_id': run_id,
                'param_config': param_config,
                'env_config': env_config,
                'param_idx': param_idx,
                'env_idx': env_idx
            })
            run_id += 1

    # Limit runs in test mode for faster execution
    if test_mode:
        # Take first environment scenario (no current) with all parameter combos
        # This gives 18 runs (3 reset × 6 leakage) for testing reset differences
        env_scenario_count = len(environment_grid)
        param_combo_count = len(parameter_grid)
        # Select first environment scenario for all parameter combos
        test_configs = []
        for param_idx in range(param_combo_count):
            run_id = param_idx * env_scenario_count  # First env scenario for each param combo
            test_configs.append(run_configs[run_id])
        run_configs = test_configs
        print(f"TEST MODE: Limited to {len(run_configs)} runs (all param combos, first env scenario)")

    # Save configuration for reference
    config_dict = {
        'waypoints': waypoints,
        'time_step': time_step,
        'control_step': control_step,
        'desire_u': desire_u,
        'total_exp_steps': total_exp_steps,
        'parameter_grid': parameter_grid,
        'environment_grid': environment_grid,
        'total_runs': len(run_configs),
        'created_at': datetime.now().isoformat()
    }

    config_path = output_dir / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Saved experiment config to: {config_path}")

    # Prepare for multiprocessing
    if test_mode:
        max_procs = 1  # Sequential for testing
    else:
        max_procs = min(12, mp.cpu_count(), len(run_configs))  # Reduced for stability

    print(f"\nRunning {len(run_configs)} simulations with {max_procs} parallel processes...")

    # Prepare function for multiprocessing
    # Create a partial function with fixed parameters for multiprocessing
    from functools import partial
    run_wrapper = partial(
        run_simulation_wrapper,
        waypoints=waypoints,
        time_step=time_step,
        total_exp_steps=total_exp_steps,
        control_step=control_step,
        desire_u=desire_u
    )

    # Run simulations
    all_results = []

    if max_procs > 1:
        with mp.Pool(processes=max_procs) as pool:
            # Use imap to preserve order and show progress
            for i, result in enumerate(pool.imap(run_wrapper, run_configs)):
                all_results.append(result)
                if (i + 1) % 5 == 0:
                    print(f"  Completed {i + 1}/{len(run_configs)} runs...")
    else:
        for i, config in enumerate(run_configs):
            result = run_wrapper(config)
            all_results.append(result)
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{len(run_configs)} runs...")

    print(f"\nAll {len(all_results)} simulations completed.")

    # Calculate scores and generate analysis
    analysis_path = output_dir / 'analysis_results.json'
    generate_analysis(all_results, analysis_path, output_dir)

    # Save results to CSV (with scores included)
    csv_path = output_dir / 'results_summary.csv'
    save_results_to_csv(all_results, csv_path)

    # Generate heatmaps
    generate_heatmaps(all_results, parameter_grid, environment_grid, output_dir)

    # Generate report
    report_path = output_dir / 'report.md'
    generate_report(all_results, analysis_path, output_dir, report_path)

    print(f"\nExperiment completed. Results saved to: {output_dir}")
    print("="*80)

    return all_results
def save_results_to_csv(results, csv_path):
    """
    Save simulation results to CSV file.

    Columns as specified in ablation_instruction.md plus additional useful fields.
    """
    fieldnames = [
        'run_id',
        'reset_mode',
        'soft_reset_coeff',
        'leakage',
        'current_speed',
        'force_magnitude',
        'direction_deg',
        'cte_rmse',
        'cte_p95',
        'score',
        'fail_reason',
        'steps',
        'finished',
        'alpha',
        'sigma'
    ]

    # Prepare rows
    rows = []
    for result in results:
        row = {field: result.get(field, '') for field in fieldnames}
        # Add alpha and sigma from leakage (sigma = leakage)
        row['alpha'] = result.get('alpha', 1.0)  # Default for no_reset
        row['sigma'] = result.get('leakage', 0.0)
        rows.append(row)

    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to CSV: {csv_path} ({len(rows)} rows)")


def generate_analysis(results, analysis_path, output_dir):
    """
    Calculate scores and analyze results.

    Scoring per ablation_instruction.md:
    1. For each environment scenario (30), normalize metrics across parameter combos (18)
    2. score = 0.7*n(cte_rmse) + 0.3*n(cte_p95)
    3. Compute mean_score and std_score for each parameter combo across all scenarios
    """
    # Group results by environment scenario
    from collections import defaultdict

    # Group by (current_speed, direction_deg)
    scenario_groups = defaultdict(list)
    for result in results:
        if result.get('is_failed', True):
            continue  # Skip failed runs for scoring
        key = (result['current_speed'], result['direction_deg'])
        scenario_groups[key].append(result)

    # For each scenario, normalize metrics and compute scores
    eps = 1e-10

    for scenario_key, scenario_results in scenario_groups.items():
        # Extract metrics
        cte_rmses = [r['cte_rmse'] for r in scenario_results if np.isfinite(r['cte_rmse'])]
        cte_p95s = [r['cte_p95'] for r in scenario_results if np.isfinite(r['cte_p95'])]

        if len(cte_rmses) == 0 or len(cte_p95s) == 0:
            continue

        # Compute min and max for normalization
        rmse_min, rmse_max = min(cte_rmses), max(cte_rmses)
        p95_min, p95_max = min(cte_p95s), max(cte_p95s)

        # Normalize and compute score for each result in this scenario
        for result in scenario_results:
            if not np.isfinite(result['cte_rmse']) or not np.isfinite(result['cte_p95']):
                result['norm_rmse'] = np.nan
                result['norm_p95'] = np.nan
                result['score'] = np.nan
                continue

            # Min-max normalization
            norm_rmse = (result['cte_rmse'] - rmse_min) / (rmse_max - rmse_min + eps)
            norm_p95 = (result['cte_p95'] - p95_min) / (p95_max - p95_min + eps)

            # Combined score (lower is better)
            score = 0.7 * norm_rmse + 0.3 * norm_p95

            result['norm_rmse'] = norm_rmse
            result['norm_p95'] = norm_p95
            result['score'] = score

    # Group by parameter combo (reset_mode, leakage)
    param_groups = defaultdict(list)
    for result in results:
        if 'score' not in result or not np.isfinite(result.get('score', np.nan)):
            continue
        key = (result['reset_mode'], result['leakage'])
        param_groups[key].append(result)

    # Compute statistics for each parameter combo
    param_stats = []
    for (reset_mode, leakage), param_results in param_groups.items():
        scores = [r['score'] for r in param_results if np.isfinite(r.get('score', np.nan))]
        if len(scores) == 0:
            continue

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Count wins (best score in each scenario)
        wins = 0
        total_scenarios = 0

        for scenario_key in scenario_groups:
            scenario_results = [r for r in scenario_groups[scenario_key]
                              if 'score' in r and np.isfinite(r.get('score', np.nan))]
            if not scenario_results:
                continue

            total_scenarios += 1
            best_score = min(r['score'] for r in scenario_results)
            # Check if this parameter combo has the best score
            for r in scenario_results:
                if r['reset_mode'] == reset_mode and r['leakage'] == leakage and r['score'] == best_score:
                    wins += 1
                    break

        win_rate = wins / total_scenarios if total_scenarios > 0 else 0

        param_stats.append({
            'reset_mode': reset_mode,
            'leakage': leakage,
            'mean_score': mean_score,
            'std_score': std_score,
            'win_rate': win_rate,
            'n_scenarios': len(scores)
        })

    # Sort by mean_score (lower is better)
    param_stats.sort(key=lambda x: x['mean_score'])

    # Save analysis results
    analysis_data = {
        'param_stats': param_stats,
        'total_scenarios': len(scenario_groups),
        'total_parameter_combos': len(param_groups),
        'scoring_weights': {'cte_rmse': 0.7, 'cte_p95': 0.3}
    }

    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)

    print(f"Saved analysis results to: {analysis_path}")

    # Print top 3 combinations
    print("\n" + "="*60)
    print("TOP 3 PARAMETER COMBINATIONS (lower mean_score is better)")
    print("="*60)
    for i, stats in enumerate(param_stats[:3]):
        print(f"{i+1}. {stats['reset_mode']} (leakage={stats['leakage']:.4f}): "
              f"mean_score={stats['mean_score']:.4f}, std={stats['std_score']:.4f}, "
              f"win_rate={stats['win_rate']:.2%}")

    return analysis_data


def generate_heatmaps(results, parameter_grid, environment_grid, output_dir):
    """
    Generate heatmaps for each parameter configuration.

    Creates heatmaps showing performance across environment scenarios.
    One heatmap per parameter combo showing cte_rmse or score across current speeds and directions.
    """
    output_dir = Path(output_dir)

    # Filter out failed runs
    valid_results = [r for r in results if not r.get('is_failed', True) and np.isfinite(r.get('cte_rmse', np.nan))]

    if not valid_results:
        print("No valid results for heatmaps")
        return

    # Organize data by parameter combo and environment
    # Structure: param_key -> (current_speed, direction_deg) -> list of cte_rmse values
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(list))

    for result in valid_results:
        key = (result['reset_mode'], result['leakage'])
        env_key = (result['current_speed'], result['direction_deg'])
        data[key][env_key].append(result['cte_rmse'])

    # Get unique current speeds and directions for axis ordering
    current_speeds = sorted(set(r['current_speed'] for r in valid_results))
    directions = sorted(set(r['direction_deg'] for r in valid_results))

    # Create heatmap for each parameter combo
    for param in parameter_grid:
        reset_mode = param['reset_mode']
        leakage = param['leakage']
        key = (reset_mode, leakage)

        if key not in data:
            continue

        # Create 2D array for heatmap
        heatmap_data = np.full((len(current_speeds), len(directions)), np.nan)

        # Fill with mean cte_rmse values
        for i, speed in enumerate(current_speeds):
            for j, direction in enumerate(directions):
                env_key = (speed, direction)
                if env_key in data[key]:
                    values = data[key][env_key]
                    if values:
                        heatmap_data[i, j] = np.mean(values)

        # Check if we have any valid data
        if np.all(np.isnan(heatmap_data)):
            continue

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap_data, cmap='viridis_r', aspect='auto')

        # Set labels
        ax.set_xticks(range(len(directions)))
        ax.set_xticklabels([f'{d}°' for d in directions])
        ax.set_yticks(range(len(current_speeds)))
        ax.set_yticklabels([f'{v:.1f} m/s' for v in current_speeds])

        # Add colorbar
        plt.colorbar(im, ax=ax, label='CTE RMSE (m)')

        # Title and labels
        title = f'Performance Heatmap: {reset_mode}, leakage={leakage:.4f}'
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Current Direction (deg)')
        ax.set_ylabel('Current Speed (m/s)')

        # Annotate cells with values
        for i in range(len(current_speeds)):
            for j in range(len(directions)):
                value = heatmap_data[i, j]
                if np.isfinite(value):
                    # Determine text color based on value relative to mean
                    valid_values = heatmap_data[np.isfinite(heatmap_data)]
                    if len(valid_values) > 0:
                        mean_val = np.mean(valid_values)
                        text_color = 'white' if value > mean_val else 'black'
                    else:
                        text_color = 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color=text_color, fontsize=8)

        plt.tight_layout()

        # Save figure
        filename = f'heatmap_{reset_mode}_leakage_{leakage:.4f}.png'.replace('.', 'p')
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved heatmap: {filename}")

    print(f"Heatmaps saved to: {output_dir}")


def generate_report(results, analysis_path, output_dir, report_path):
    """
    Generate markdown report with experiment summary and findings.
    """
    # Load analysis data
    with open(analysis_path, 'r') as f:
        analysis_data = json.load(f)

    param_stats = analysis_data['param_stats']

    # Create report content
    report_lines = []

    report_lines.append("# ALOS Parameter Ablation Experiment Report")
    report_lines.append("")
    report_lines.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    report_lines.append("")

    report_lines.append("## Experiment Setup")
    report_lines.append("")
    report_lines.append("### Parameter Grid")
    report_lines.append("- **Reset mechanisms**: hard_reset (α=0.0), soft_reset (α=0.5), no_reset (α=1.0)")
    report_lines.append("- **Leakage values**: 0.0, 0.1, 0.01, 0.005, 0.003, 0.001")
    report_lines.append("- **Total parameter combos**: 18")
    report_lines.append("")

    report_lines.append("### Environment Scenarios")
    report_lines.append("- **Current speeds**: 0.1, 0.2, 0.3, 0.4 m/s")
    report_lines.append("- **Directions**: 0°, 30°, 90°, 150°, 180°, 270°")
    report_lines.append("- **Force magnitude**: (current_speed / 0.1) × 5 N")
    report_lines.append("- **Total scenarios**: 24")
    report_lines.append("")

    report_lines.append("### Fixed Configuration")
    report_lines.append("- **Clamp**: ON (β_max = 30°)")
    report_lines.append("- **Conditional integration**: ON (ψ_err_threshold = 15°)")
    report_lines.append("- **Path**: Zigzag [(0,0), (100,100), (0,200), (100,300), (0,400)]")
    report_lines.append("- **Total simulations**: 432")
    report_lines.append("")

    report_lines.append("### Scoring")
    report_lines.append("For each environment scenario:")
    report_lines.append("1. Normalize CTE RMSE and CTE P95 across parameter combos")
    report_lines.append("2. Combined score: `0.7 × norm(CTE RMSE) + 0.3 × norm(CTE P95)`")
    report_lines.append("3. Lower score is better")
    report_lines.append("")

    report_lines.append("## Results")
    report_lines.append("")

    # Top combinations table
    report_lines.append("### Top 3 Parameter Combinations")
    report_lines.append("")
    report_lines.append("| Rank | Reset Mode | Leakage | Mean Score | Std Score | Win Rate |")
    report_lines.append("|------|------------|---------|------------|-----------|----------|")

    for i, stats in enumerate(param_stats[:3]):
        report_lines.append(f"| {i+1} | {stats['reset_mode']} | {stats['leakage']:.4f} | "
                          f"{stats['mean_score']:.4f} | {stats['std_score']:.4f} | "
                          f"{stats['win_rate']:.2%} |")

    report_lines.append("")

    # Best combination details
    if param_stats:
        best = param_stats[0]
        report_lines.append(f"### Optimal Combination")
        report_lines.append("")
        report_lines.append(f"- **Reset mechanism**: {best['reset_mode']}")
        report_lines.append(f"- **Leakage**: {best['leakage']:.4f}")
        if best['reset_mode'] == 'soft_reset':
            report_lines.append(f"- **Soft reset coefficient**: 0.5")
        report_lines.append(f"- **Mean score**: {best['mean_score']:.4f}")
        report_lines.append(f"- **Score std**: {best['std_score']:.4f}")
        report_lines.append(f"- **Win rate**: {best['win_rate']:.2%} (wins in {best['win_rate']*analysis_data['total_scenarios']:.0f} of {analysis_data['total_scenarios']} scenarios)")
        report_lines.append("")

    # Full results table (abbreviated)
    report_lines.append("### All Parameter Combinations (Sorted)")
    report_lines.append("")
    report_lines.append("| Reset Mode | Leakage | Mean Score | Std Score | Win Rate |")
    report_lines.append("|------------|---------|------------|-----------|----------|")

    for stats in param_stats[:10]:  # Show top 10
        report_lines.append(f"| {stats['reset_mode']} | {stats['leakage']:.4f} | "
                          f"{stats['mean_score']:.4f} | {stats['std_score']:.4f} | "
                          f"{stats['win_rate']:.2%} |")

    if len(param_stats) > 10:
        report_lines.append(f"| ... {len(param_stats)-10} more combinations ... |")

    report_lines.append("")

    # Output files
    report_lines.append("## Output Files")
    report_lines.append("")
    report_lines.append(f"- `results_summary.csv`: Raw simulation results ({len(results)} rows)")
    report_lines.append(f"- `analysis_results.json`: Analysis data")
    report_lines.append(f"- `heatmap_*.png`: Performance heatmaps for each parameter combo")
    report_lines.append(f"- `experiment_config.json`: Experiment configuration")
    report_lines.append(f"- `report.md`: This report")
    report_lines.append("")

    # Failure Analysis
    report_lines.append("## Failure Analysis")
    report_lines.append("")
    
    # Load experiment config to get total_exp_steps
    config_path = output_dir / 'experiment_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        total_exp_steps = config.get('total_exp_steps', 300000)
    else:
        total_exp_steps = 300000
    
    # Collect failed runs (finished=False or fail_reason not empty)
    failed_runs = []
    for r in results:
        if not r.get('finished', False) or r.get('fail_reason', '').strip():
            failed_runs.append(r)
    
    if failed_runs:
        # Limit to at most 5
        display_runs = failed_runs[:5]
        report_lines.append(f"**{len(failed_runs)}** out of {len(results)} runs failed or did not complete.")
        report_lines.append("")
        report_lines.append("Top failed configurations:")
        report_lines.append("")
        report_lines.append("| Reset Mode | Leakage | Current | Force | Direction | Fail Reason |")
        report_lines.append("|------------|---------|---------|-------|-----------|-------------|")
        for r in display_runs:
            reset = r.get('reset_mode', 'hard_reset')
            leakage = r.get('leakage', 0.0)
            current = r.get('current_speed', 0.0)
            force = r.get('force_magnitude', 0.0)
            direction = r.get('direction_deg', 0)
            reason = r.get('fail_reason', 'unknown').replace('max_steps_exceeded', 'max steps exceeded')
            report_lines.append(f"| {reset} | {leakage} | {current} m/s | {force} N | {direction}° | {reason} |")
        report_lines.append("")
        report_lines.append("To visualize these failed cases, run:")
        report_lines.append("```bash")
        for r in display_runs:
            reset_mode = r.get('reset_mode', 'hard_reset')
            if reset_mode == 'hard_reset':
                alpha = 0.0
            elif reset_mode == 'soft_reset':
                alpha = 0.5
            else:  # no_reset
                alpha = 1.0
            leakage = r.get('leakage', 0.0)
            current = r.get('current_speed', 0.0)
            force = r.get('force_magnitude', 0.0)
            direction = r.get('direction_deg', 0)
            report_lines.append(f"python demos/alos/demo_alos_single_scenario.py --reset {alpha} --leakage {leakage} --force {force} --current {current} --direction {direction} --max-steps {total_exp_steps}")
        report_lines.append("```")
    else:
        report_lines.append("All runs completed successfully.")
    
    report_lines.append("")

    # Conclusion
    report_lines.append("## Conclusion")
    report_lines.append("")
    if param_stats:
        best = param_stats[0]
        report_lines.append(f"The optimal ALOS parameter combination for the tested zigzag path under varying current+force disturbances is **{best['reset_mode']} with leakage={best['leakage']:.4f}**.")
        report_lines.append(f"This combination achieved the lowest mean score ({best['mean_score']:.4f}) with reasonable robustness (std={best['std_score']:.4f}).")
    else:
        report_lines.append("No valid results were obtained from the simulations.")
    report_lines.append("")

    # Write report
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Generated report: {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ALOS parameter ablation experiment"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: exp_data/alos_param_ablation_TIMESTAMP)'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with reduced steps'
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"exp_data/alos_param_ablation_half_scale_{timestamp}")

    # Run experiment
    run_experiment_parallel(output_dir, test_mode=args.test_mode)
