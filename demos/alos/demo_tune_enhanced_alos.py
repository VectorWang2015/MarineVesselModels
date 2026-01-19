import numpy as np
import itertools
import os

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


def evaluate_parameters(beta_max_deg, psi_err_threshold_deg, alpha, scenario='zigzag', guider_class=EnhancedAdaptiveLOSGuider):
    """
    Run simulation with given enhancement parameters and return performance metrics.
    """
    # Simulation parameters (same as demo)
    time_step = 0.05
    control_step = 0.1
    control_every = int(control_step / time_step)
    total_exp_steps = 100 if os.environ.get('TEST_MODE', '0') == '1' else 10000
    desire_u = 0.5
    max_base_N = 1000
    max_diff_N = 2000
    env_force_magnitude = 200.0
    env_force_direction = np.deg2rad(60.0)
    forward_dist = 5.0
    reached_threshold = 2.0
    gamma = 0.005
    beta_hat0 = 0.0
    
    # Zigzag waypoints (same as demo)
    if scenario == 'zigzag':
        waypoints = [
            (0, 0),
            (100, 100),
            (200, 0),
            (300, 100),
            (400, 0)
        ]
    else:
        # Straight line (optional)
        waypoints = [(0, 0), (400, 0)]
    
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

    # Build guider kwargs
    guider_kwargs = {
        'waypoints': waypoints,
        'reached_threshold': reached_threshold,
        'output_err_flag': False,
        'forward_dist': forward_dist,
        'gamma': gamma,
        'beta_hat0': beta_hat0,
        'dt': control_step,
    }
    if guider_class == EnhancedAdaptiveLOSGuider:
        guider_kwargs.update({
            'beta_max': np.deg2rad(beta_max_deg),
            'psi_err_threshold': np.deg2rad(psi_err_threshold_deg),
            'alpha': alpha
        })
    
    guider = guider_class(**guider_kwargs)

    result = test_guider_with_beta(simulator, thruster, diff_controller, u_controller, guider,
                                   current_state, total_exp_steps, control_every, time_step,
                                   desire_u, pose_draw_interval=None)
    
    # Compute metrics
    y_es = np.array(result['y_es'])
    mean_abs_err = np.mean(np.abs(y_es))
    max_abs_err = np.max(np.abs(y_es))
    beta_hats = np.array(result['beta_hats'])
    mean_beta = np.mean(beta_hats)
    std_beta = np.std(beta_hats)
    max_beta = np.max(np.abs(beta_hats))
    limit = np.deg2rad(beta_max_deg)  # beta_max limit for counting hits
    hit_limit = np.sum(np.abs(beta_hats) >= limit * 0.99)
    total = len(beta_hats)
    
    return {
        'mean_abs_err': mean_abs_err,
        'max_abs_err': max_abs_err,
        'mean_beta': mean_beta,
        'std_beta': std_beta,
        'max_beta': max_beta,
        'hit_limit': hit_limit,
        'total': total,
        'steps': result['steps']
    }


def main():
    # Parameter grids
    beta_max_vals = [15, 30, 45, 60]  # degrees
    psi_err_threshold_vals = [5, 10, 15, 20, 30]  # degrees
    alpha_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Baseline original ALOS (no enhancements)
    baseline_params = {'beta_max_deg': 999, 'psi_err_threshold_deg': 999, 'alpha': 0.0}
    print("Running baseline (original ALOS)...")
    baseline_metrics = evaluate_parameters(**baseline_params)
    print(f"Baseline - Mean |error|: {baseline_metrics['mean_abs_err']:.3f} m, "
          f"Std β̂: {np.rad2deg(baseline_metrics['std_beta']):.2f}°")
    
    results = []
    
    # Iterate over all combinations
    total_combos = len(beta_max_vals) * len(psi_err_threshold_vals) * len(alpha_vals)
    print(f"\nTesting {total_combos} parameter combinations...")
    
    for i, (beta_max, psi_err_threshold, alpha) in enumerate(
        itertools.product(beta_max_vals, psi_err_threshold_vals, alpha_vals), 1
    ):
        print(f"  [{i}/{total_combos}] beta_max={beta_max}°, "
              f"psi_err_threshold={psi_err_threshold}°, alpha={alpha}")
        
        metrics = evaluate_parameters(beta_max, psi_err_threshold, alpha)
        
        results.append({
            'beta_max': beta_max,
            'psi_err_threshold': psi_err_threshold,
            'alpha': alpha,
            'mean_abs_err': metrics['mean_abs_err'],
            'max_abs_err': metrics['max_abs_err'],
            'std_beta': metrics['std_beta'],
            'max_beta': metrics['max_beta'],
            'hit_limit': metrics['hit_limit'],
            'steps': metrics['steps']
        })
    
    # Sort by mean absolute error (primary metric)
    results_sorted = sorted(results, key=lambda x: x['mean_abs_err'])
    
    # Print top 10 configurations
    print("\n" + "="*80)
    print("TOP 10 PARAMETER CONFIGURATIONS (by mean absolute cross-track error)")
    print("="*80)
    print("Rank | beta_max | psi_err_thresh | alpha | Mean |error| (m) | Std β̂ (°) | Max |error| (m) | Steps")
    print("-"*80)
    for rank, r in enumerate(results_sorted[:10], 1):
        print(f"{rank:4} | {r['beta_max']:8} | {r['psi_err_threshold']:15} | {r['alpha']:5} | "
              f"{r['mean_abs_err']:19.3f} | {np.rad2deg(r['std_beta']):10.2f} | "
              f"{r['max_abs_err']:16.3f} | {r['steps']:5}")
    
    # Compare with baseline
    baseline_rank = None
    for rank, r in enumerate(results_sorted, 1):
        if (r['beta_max'] == 999 and r['psi_err_threshold'] == 999 and r['alpha'] == 0.0):
            baseline_rank = rank
            break
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON")
    print(f"Baseline (original ALOS) rank: {baseline_rank if baseline_rank else 'N/A'}")
    print(f"Baseline mean |error|: {baseline_metrics['mean_abs_err']:.3f} m")
    print(f"Best enhanced mean |error|: {results_sorted[0]['mean_abs_err']:.3f} m")
    improvement = (baseline_metrics['mean_abs_err'] - results_sorted[0]['mean_abs_err']) / baseline_metrics['mean_abs_err'] * 100
    print(f"Improvement: {improvement:.1f}%")
    print("="*80)
    
    # Save results to file
    import json
    with open('enhanced_alos_tuning_results.json', 'w') as f:
        json.dump({
            'baseline': baseline_metrics,
            'results': results,
            'sorted_indices': [results.index(r) for r in results_sorted]
        }, f, indent=2)
    print("Results saved to enhanced_alos_tuning_results.json")


if __name__ == "__main__":
    main()