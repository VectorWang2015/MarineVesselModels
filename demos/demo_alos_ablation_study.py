"""
Ablation study comparing Dynamic LOS, Adaptive LOS, and Enhanced ALOS variants.

Enhanced ALOS variants:
1. Clamp + Conditional integration only (α=0.0, hard reset)
2. Clamp + Partial reset only (ψ_err_thresh=999°, disable conditional)
3. All enhancements (clamp + conditional + partial reset)

Scenarios:
- Zigzag path (400m)
- Square trajectory (400m perimeter)
- Custom zigzag path (used in other demos)

Disturbance levels: 5N, 16N, 40N at 60° direction

Metrics:
- Mean absolute cross-track error (m)
- RMSE of cross-track error (m)

Trajectory plots are generated for High disturbance (40N) with environmental
force direction arrows shown in background.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

import plot_utils

from MarineVesselModels.simulator import SimplifiedEnvironmentalDisturbanceSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, Fossen

from control.pid import DoubleLoopHeadingPID, PIDAW
from control.los import DynamicDistLOSGuider, AdaptiveLOSGuider, EnhancedAdaptiveLOSGuider


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
                             desire_u, record_trajectory=False):
    """
    Run guidance simulation and collect cross-track error metrics.

    :param simulator: Vessel simulator
    :param thruster: Thruster model
    :param diff_controller: Heading controller
    :param u_controller: Velocity controller
    :param guider: LOS guider instance
    :param current_state: Initial state vector
    :param total_exp_steps: Maximum simulation steps
    :param control_every: Control period in steps
    :param time_step: Simulation time step (seconds)
    :param desire_u: Desired surge velocity (m/s)
    :param record_trajectory: If True, record x,y trajectory
    :return: Dictionary with metrics and raw data
    """
    y_es = []  # cross-track errors at each simulation step
    xs = []  # x positions (north)
    ys = []  # y positions (east)
    is_ended = False

    # Initialize control variables
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

        # Control every control_step seconds
        if t % control_every == 0:
            is_ended, desired_psi = guider.step((current_x, current_y), current_psi)
            if is_ended:
                break
            # Calculate control signals
            diff_control_signal, ref_r = diff_controller.step(
                psi_ref=desired_psi, psi=current_psi, r=current_r)
            velo_control_signal = u_controller.step(sp=desire_u, y=current_u)

            psi_err = ((desired_psi - current_psi) + np.pi) % (2*np.pi) - np.pi

            left = velo_control_signal + diff_control_signal
            right = velo_control_signal - diff_control_signal
            tau = thruster.newton_to_tau(left, right)

        # Record cross-track error (at each simulation step)
        y_es.append(compute_cross_track_error(guider))
        
        # Record trajectory if requested
        if record_trajectory:
            xs.append(current_x)
            ys.append(current_y)

        # Apply control and step simulation
        current_state = simulator.step(tau)

    # Compute metrics
    y_es_array = np.array(y_es)
    mean_abs_err = np.mean(np.abs(y_es_array)) if len(y_es_array) > 0 else np.nan
    max_abs_err = np.max(np.abs(y_es_array)) if len(y_es_array) > 0 else np.nan
    rmse = np.sqrt(np.mean(y_es_array**2)) if len(y_es_array) > 0 else np.nan

    return {
        'mean_abs_err': mean_abs_err,
        'max_abs_err': max_abs_err,
        'rmse': rmse,
        'steps': len(y_es),
        'finished_early': is_ended,
        'y_es': y_es_array,
        'xs': np.array(xs) if record_trajectory else None,
        'ys': np.array(ys) if record_trajectory else None,
        'record_trajectory': record_trajectory,
    }


if __name__ == "__main__":
    import sys
    
    # Check if TEST_MODE is enabled
    TEST_MODE = os.environ.get('TEST_MODE', '0') == '1'
    
    # Simulation parameters
    time_step = 0.1
    control_step = 0.2
    control_every = int(control_step / time_step)
    total_exp_steps = 100 if TEST_MODE else 15000  # Reduced for testing
    
    desire_u = 0.5  # m/s
    max_base_N = 100.0
    max_diff_N = 60.0
    
    forward_dist = 5.0
    reached_threshold = 5.0
    
    # Environmental disturbance direction (fixed at 60°)
    env_force_direction = np.deg2rad(60.0)
    
    # Define scenarios
    scenarios = [
        {
            'name': 'Zigzag',
            'waypoints': [(0, 0), (100, 100), (200, 0), (300, 100), (400, 0)],
            'description': 'Zigzag path, 400m total length'
        },
        {
            'name': 'Square',
            'waypoints': [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)],
            'description': 'Square trajectory, 400m perimeter'
        },
        {
            'name': 'Custom Zigzag',
            'waypoints': [(60, 15), (0, 30), (45, 45), (0, 60), (30, 75), (0, 90), (15, 105), (0, 120)],
            'description': 'Custom zigzag path used in other demos'
        }
    ]
    
    # Disturbance levels (N)
    disturbances = [
        {'name': 'Low', 'magnitude': 5.0},
        {'name': 'Medium', 'magnitude': 16.0},
        {'name': 'High', 'magnitude': 40.0}
    ]
    
    # LOS guider configurations
    guider_configs = [
        {
            'name': 'Dynamic LOS',
            'class': DynamicDistLOSGuider,
            'kwargs': {'output_err_flag': False, 'forward_dist': forward_dist}
        },
        {
            'name': 'Adaptive LOS',
            'class': AdaptiveLOSGuider,
            'kwargs': {
                'output_err_flag': False,
                'forward_dist': forward_dist,
                'gamma': 0.005,
                'beta_hat0': 0.0,
                'dt': control_step
            }
        },
        {
            'name': 'Enhanced (Clamp+Cond)',
            'class': EnhancedAdaptiveLOSGuider,
            'kwargs': {
                'output_err_flag': False,
                'forward_dist': forward_dist,
                'gamma': 0.005,
                'beta_hat0': 0.0,
                'dt': control_step,
                'beta_max': np.deg2rad(30.0),
                'psi_err_threshold': np.deg2rad(15.0),
                'alpha': 0.0  # Hard reset (clamp + conditional only)
            }
        },
        {
            'name': 'Enhanced (Clamp+Partial)',
            'class': EnhancedAdaptiveLOSGuider,
            'kwargs': {
                'output_err_flag': False,
                'forward_dist': forward_dist,
                'gamma': 0.005,
                'beta_hat0': 0.0,
                'dt': control_step,
                'beta_max': np.deg2rad(30.0),
                'psi_err_threshold': np.deg2rad(999.0),  # Effectively disable conditional
                'alpha': 0.5  # Partial reset
            }
        },
        {
            'name': 'Enhanced (All)',
            'class': EnhancedAdaptiveLOSGuider,
            'kwargs': {
                'output_err_flag': False,
                'forward_dist': forward_dist,
                'gamma': 0.005,
                'beta_hat0': 0.0,
                'dt': control_step,
                'beta_max': np.deg2rad(30.0),
                'psi_err_threshold': np.deg2rad(15.0),
                'alpha': 0.5  # Full enhancement
            }
        }
    ]
    
    print("="*80)
    print("ABLATION STUDY: LOS GUIDER COMPARISON")
    print("="*80)
    print(f"Test mode: {'ON (100 steps per sim)' if TEST_MODE else 'OFF (full simulations)'}")
    print(f"Total simulations: {len(scenarios)} scenarios × {len(disturbances)} disturbances × {len(guider_configs)} guiders = {len(scenarios)*len(disturbances)*len(guider_configs)}")
    print("="*80)
    
    # Store results
    results = {}
    # Store trajectory data for Medium disturbance only (for plotting)
    trajectories = {}
    
    # Run simulations
    total_sims = len(scenarios) * len(disturbances) * len(guider_configs)
    sim_count = 0
    
    for scenario in scenarios:
        scenario_name = scenario['name']
        waypoints = scenario['waypoints']
        results[scenario_name] = {}
        # Initialize trajectory storage for this scenario
        trajectories[scenario_name] = {
            'waypoints': waypoints,
            'data': {}  # will store guider_name -> trajectory data for High disturbance
        }
        
        for disturbance in disturbances:
            dist_name = disturbance['name']
            env_force_magnitude = disturbance['magnitude']
            results[scenario_name][dist_name] = {}
            
            print(f"\nScenario: {scenario_name}, Disturbance: {dist_name} ({env_force_magnitude} N)")
            print("-"*60)
            
            for guider_config in guider_configs:
                guider_name = guider_config['name']
                sim_count += 1
                
                print(f"  [{sim_count}/{total_sims}] Testing {guider_name}...", end='', flush=True)
                
                # Initialize simulator and controllers
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
                
                # Create guider instance
                guider = guider_config['class'](
                    waypoints=waypoints,
                    reached_threshold=reached_threshold,
                    **guider_config['kwargs']
                )
                
                # Determine if we should record trajectory (High disturbance only)
                record_traj = (dist_name == 'High')
                
                # Run simulation
                result = test_guider_with_metrics(
                    simulator, thruster, diff_controller, u_controller, guider,
                    current_state, total_exp_steps, control_every, time_step,
                    desire_u, record_trajectory=record_traj
                )
                
                # Store results
                results[scenario_name][dist_name][guider_name] = {
                    'mean_abs_err': result['mean_abs_err'],
                    'max_abs_err': result['max_abs_err'],
                    'rmse': result['rmse'],
                    'steps': result['steps'],
                    'finished': result['finished_early']
                }
                
                # Store trajectory data if recorded
                if record_traj and result['xs'] is not None:
                    trajectories[scenario_name]['data'][guider_name] = {
                        'xs': result['xs'],
                        'ys': result['ys']
                    }
                
                print(f" done. Steps: {result['steps']}, Mean|error|: {result['mean_abs_err']:.3f} m")
    
    # ============================================================================
    # Plot trajectories for High disturbance
    # ============================================================================
    if trajectories:
        # Colors for different guiders
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(guider_configs)))
        
        for scenario_name, scenario_data in trajectories.items():
            waypoints = scenario_data['waypoints']
            traj_data = scenario_data['data']
            if not traj_data:
                continue  # No trajectory data for this scenario
                
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot waypoints
            wp_x = [pt[0] for pt in waypoints]
            wp_y = [pt[1] for pt in waypoints]
            ax.scatter(wp_y, wp_x, marker='x', color='black', s=100, label='Waypoints')
            ax.plot(wp_y, wp_x, '--', color='black', alpha=0.5)
            
            # Collect all trajectory points to determine axis limits
            all_xs = []
            all_ys = []
            for guider_name, data in traj_data.items():
                xs = data['xs']
                ys = data['ys']
                if xs is None or ys is None or len(xs) == 0:
                    continue
                all_xs.extend(xs)
                all_ys.extend(ys)
            
            # Calculate axis limits with margin
            margin = 20.0
            if all_xs and all_ys:
                x_min = min(min(wp_x), min(all_xs)) - margin
                x_max = max(max(wp_x), max(all_xs)) + margin
                y_min = min(min(wp_y), min(all_ys)) - margin
                y_max = max(max(wp_y), max(all_ys)) + margin
            else:
                # Fallback to waypoint-only limits
                x_min = min(wp_x) - margin
                x_max = max(wp_x) + margin
                y_min = min(wp_y) - margin
                y_max = max(wp_y) + margin
            
            ax.set_xlim(y_min, y_max)
            ax.set_ylim(x_min, x_max)
            ax.set_aspect('equal')
            
            # Add background arrows showing environmental force direction (60° in NED)
            plot_utils.add_force_direction_arrows(
                ax=ax,
                direction_angle=env_force_direction,
                spacing=10.0,
            )
            
            # Plot each guider's trajectory
            for idx, (guider_name, data) in enumerate(traj_data.items()):
                xs = data['xs']
                ys = data['ys']
                if xs is None or ys is None or len(xs) == 0:
                    continue
                ax.plot(ys, xs, color=colors[idx], linewidth=2, alpha=0.9, label=guider_name)
            
            ax.set_xlabel('Y position (m)')
            ax.set_ylabel('X position (m)')
            ax.set_title(f'{scenario_name} - Trajectories (High disturbance, 40N at 60°)')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            fig.tight_layout()
            
            # Save or show figure based on backend
            import matplotlib
            if matplotlib.get_backend().lower() == 'agg':
                # Non-interactive backend: save figure
                filename = f'ablation_trajectory_{scenario_name.replace(" ", "_")}_high_dist.png'
                fig.savefig(filename, dpi=150)
                print(f"Saved trajectory plot: {filename}")
                plt.close(fig)
            else:
                # Interactive backend: show figure
                plt.show()
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Generate markdown tables for each scenario
    for scenario_name in results:
        print(f"\n## Scenario: {scenario_name}")
        print("\n| Guider | Disturbance | Mean |error| (m) | RMSE (m) | Steps | Finished |")
        print("|--------|-------------|------------------|----------|--------|----------|")
        
        for dist_name in results[scenario_name]:
            for guider_name in results[scenario_name][dist_name]:
                r = results[scenario_name][dist_name][guider_name]
                finished_str = "✓" if r['finished'] else "✗"
                print(f"| {guider_name} | {dist_name} | {r['mean_abs_err']:.3f} | {r['rmse']:.3f} | {r['steps']} | {finished_str} |")
        
        # Add separator between scenarios
        print("\n---")
    
    # Generate comparison table per disturbance level
    print("\n" + "="*80)
    print("PER-DISTURBANCE COMPARISON (Mean |error|)")
    print("="*80)
    
    for disturbance in disturbances:
        dist_name = disturbance['name']
        print(f"\n### Disturbance: {dist_name}")
        
        # Build header
        header = "| Guider |"
        separator = "|--------|"
        for scenario in scenarios:
            header += f" {scenario['name']} (m) |"
            separator += "------------|"
        print("\n" + header)
        print(separator)
        
        # Rows for each guider
        for guider_config in guider_configs:
            guider_name = guider_config['name']
            row = f"| {guider_name} |"
            for scenario in scenarios:
                scenario_name = scenario['name']
                val = results[scenario_name][dist_name][guider_name]['mean_abs_err']
                row += f" {val:.3f} |"
            print(row)
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)