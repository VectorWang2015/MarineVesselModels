#!/usr/bin/env python3
"""
Test interaction between environmental force (with N_d moment) and ocean current.

This demo tests the newly modified SimplifiedEnvironmentalDisturbanceSimulator
which computes N_d moment from environmental force Y_d using l_eff.

Features:
1. Fixed current: 0.1 m/s at 0° (North)
2. Fixed environmental force: 5 N at 0° (same direction as current)
3. Tests four scenarios:
   - Drift with current only (no force, no control)
   - Drift with current + environmental force (no control)
   - Circle movement with current only (25N, 75N thrust)
   - Circle movement with current + environmental force (25N, 75N thrust)
4. l_eff = 0.1 m for moment calculation
5. Plot shows all scenarios with force/current arrows in background
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle as MplCircle

from MarineVesselModels.simulator import SimplifiedEnvironmentalDisturbanceSimulator
from MarineVesselModels.Fossen import sample_hydro_params_2, sample_b_2, FossenWithCurrent
from MarineVesselModels.thrusters import NaiveDoubleThruster
from plot_utils import add_force_direction_arrows, draw_ship_pose_ned

# Scenario labels for plotting
SCENARIO_LABELS = {
    'drift_current_only': 'Drift (current only)',
    'drift_current_force': 'Drift (current + force)',
    'circle_current_only': 'Circle (current only)',
    'circle_current_force': 'Circle (current + force)'
}

# Scenario colors for plotting - red group for current-only, blue group for current+force
SCENARIO_COLORS = {
    'drift_current_only': '#FF0000',  # pink red
    'drift_current_force': '#0000FF',  # light blue
    'circle_current_only': '#FF0000',  # 255 red
    'circle_current_force': '#0000FF'   # 255 blue
}


def run_scenario(
    scenario_name,
    current_velocity,
    current_direction,
    env_force_magnitude,
    env_force_direction,
    l_eff,
    control_tau,
    time_step,
    total_steps
):
    """
    Run a single simulation scenario.

    Args:
        scenario_name: Name for debugging/output
        current_velocity: Current magnitude in m/s
        current_direction: Current direction in radians (NED coordinates)
        env_force_magnitude: Environmental force magnitude in N
        env_force_direction: Environmental force direction in radians (NED coordinates)
        l_eff: Effective arm for moment calculation (m)
        control_tau: Control input torque vector (3x1)
        time_step: Simulation time step (s)
        total_steps: Number of simulation steps

    Returns:
        tuple: (xs, ys, psis, simulator) - trajectory, heading history, and simulator object
    """
    print(f"  Running {scenario_name}...")

    simulator = SimplifiedEnvironmentalDisturbanceSimulator(
        hydro_params=sample_hydro_params_2,
        time_step=time_step,
        model=FossenWithCurrent,
        env_force_magnitude=env_force_magnitude,
        env_force_direction=env_force_direction,
        l_eff=l_eff,
        current_velocity=current_velocity,
        current_direction=current_direction,
    )

    xs, ys, psis = [], [], []

    for _ in range(total_steps):
        # Get current state
        state = simulator.state
        x = state[0][0]  # North position
        y = state[1][0]  # East position
        psi = state[2][0]  # Heading

        xs.append(x)
        ys.append(y)
        psis.append(psi)

        # Apply control input (may be zero for drift scenarios)
        simulator.step(control_tau)

    return np.array(xs), np.array(ys), np.array(psis), simulator


def plot_results(
    results_dict,
    current_velocity,
    current_direction,
    env_force_magnitude,
    env_force_direction,
    l_eff,
    time_step,
    total_steps
):
    """
    Create optimized plot showing all scenarios in a single axis.
    """
    # Create figure with single axis
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f'Environmental Force + Current Interaction\n'
                 f'Current: {current_velocity} m/s at {np.rad2deg(current_direction):.0f}°, '
                 f'Force: {env_force_magnitude} N at {np.rad2deg(env_force_direction):.0f}°, '
                 f'l_eff: {l_eff} m', fontsize=14)

    # Use module-level constants for colors and labels
    colors = SCENARIO_COLORS
    labels = SCENARIO_LABELS

    # Plot trajectories with reduced linewidth
    for scenario, (xs, ys, _) in results_dict.items():
        # Determine alpha and linestyle based on scenario
        if 'drift' in scenario:
            alpha = 0.5
            linestyle = '--'
        else:
            alpha = 0.9
            linestyle = '-'
        linewidth = 1.5
        ax.plot(ys, xs, color=colors[scenario], linewidth=linewidth, alpha=alpha, label=labels[scenario], linestyle=linestyle)

        # Add end point with appropriate marker
        final_x = xs[-1]
        final_y = ys[-1]
        if 'force' in scenario:
            marker = 'o'  # circle for current+force
        else:
            marker = 'x'  # x for current only
        ax.scatter(final_y, final_x, color=colors[scenario], marker=marker, s=80, zorder=5, alpha=alpha)

    # Add direction arrows (only one since current and force share same direction)
    # Use default color from plot_utils function
    add_force_direction_arrows(
        ax=ax,
        direction_angle=current_direction,
        spacing=1.0,
        alpha=0.25,
        coord_system='NED'
    )

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Add text annotation about current and force
    """
    current_text = f'Current: {current_velocity} m/s at {np.rad2deg(current_direction):.0f}°'
    force_text = f'Force: {env_force_magnitude} N at {np.rad2deg(env_force_direction):.0f}°'
    l_eff_text = f'l_eff: {l_eff} m (N_d = Y_d × l_eff)'

    info_text = f'{current_text}\n{force_text}\n{l_eff_text}'
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)
    """

    plt.tight_layout()

    # Show interactive plot (non-blocking for CLI use)
    import matplotlib
    if matplotlib.get_backend().lower() == 'agg':
        print("  Note: Non-interactive backend. Figure not displayed.")
        plt.close()
    else:
        plt.show(block=False)

def main():
    """Main function to run the demo."""
    print("=" * 80)
    print("Environmental Force + Current Interaction Demo")
    print("Testing N_d moment calculation from environmental force Y_d using l_eff")
    print("=" * 80)

    # Simulation parameters
    time_step = 0.1
    total_steps = 1000  # 100 seconds

    # Current settings (fixed)
    current_velocity = 0.1  # m/s
    current_direction = 0.0  # radians (0° = North in NED)

    # Environmental force settings (fixed)
    env_force_magnitude = 5.0  # N
    env_force_direction = 0.0  # radians (same direction as current)
    l_eff = 0.1  # meters (effective arm for moment calculation)

    # Control configurations
    thruster = NaiveDoubleThruster(b=sample_b_2)

    # Zero thrust for drift scenarios
    zero_tau = thruster.newton_to_tau(l_thrust_N=0.0, r_thrust_N=0.0)

    # Asymmetric thrust for circle movement (25N left, 75N right)
    circle_tau = thruster.newton_to_tau(l_thrust_N=25.0, r_thrust_N=75.0)

    # Define scenarios to test
    scenarios = [
        {
            'name': 'drift_current_only',
            'current_velocity': current_velocity,
            'current_direction': current_direction,
            'env_force_magnitude': 0.0,  # No environmental force
            'env_force_direction': 0.0,
            'l_eff': l_eff,
            'control_tau': zero_tau,
            'description': 'Drift with current only (no force, no control)'
        },
        {
            'name': 'drift_current_force',
            'current_velocity': current_velocity,
            'current_direction': current_direction,
            'env_force_magnitude': env_force_magnitude,
            'env_force_direction': env_force_direction,
            'l_eff': l_eff,
            'control_tau': zero_tau,
            'description': 'Drift with current + environmental force (no control)'
        },
        {
            'name': 'circle_current_only',
            'current_velocity': current_velocity,
            'current_direction': current_direction,
            'env_force_magnitude': 0.0,  # No environmental force
            'env_force_direction': 0.0,
            'l_eff': l_eff,
            'control_tau': circle_tau,
            'description': 'Circle movement with current only (25N, 75N thrust)'
        },
        {
            'name': 'circle_current_force',
            'current_velocity': current_velocity,
            'current_direction': current_direction,
            'env_force_magnitude': env_force_magnitude,
            'env_force_direction': env_force_direction,
            'l_eff': l_eff,
            'control_tau': circle_tau,
            'description': 'Circle movement with current + environmental force (25N, 75N thrust)'
        }
    ]

    # Run all scenarios
    print(f"\nRunning {len(scenarios)} scenarios:")
    for scenario in scenarios:
        print(f"  - {scenario['description']}")

    results_dict = {}
    simulators_dict = {}

    for scenario in scenarios:
        xs, ys, psis, simulator = run_scenario(
            scenario_name=scenario['description'],
            current_velocity=scenario['current_velocity'],
            current_direction=scenario['current_direction'],
            env_force_magnitude=scenario['env_force_magnitude'],
            env_force_direction=scenario['env_force_direction'],
            l_eff=scenario['l_eff'],
            control_tau=scenario['control_tau'],
            time_step=time_step,
            total_steps=total_steps
        )

        simulators_dict[scenario['name']] = simulator
        results_dict[scenario['name']] = (xs, ys, psis)

    # Moment validation
    print("\n" + "=" * 80)
    print("MOMENT VALIDATION")
    print("=" * 80)
    for scenario_name, simulator in simulators_dict.items():
        if simulator.env_force_magnitude > 0:
            env_forces_body = simulator.env_forces_body
            errors = []
            for force in env_forces_body:
                Y_d = force[1][0]
                N_d = force[2][0]
                expected_N_d = Y_d * simulator.l_eff
                errors.append(abs(N_d - expected_N_d))
            max_error = np.max(errors) if errors else 0
            avg_error = np.mean(errors) if errors else 0
            print(f"\n{SCENARIO_LABELS[scenario_name]}:")
            print(f"  l_eff: {simulator.l_eff} m")
            print(f"  Max |N_d - Y_d*l_eff|: {max_error:.6e} N")
            print(f"  Avg |N_d - Y_d*l_eff|: {avg_error:.6e} N")
            if max_error < 1e-10:
                print(f"  ✓ Moment calculation correct")
            else:
                print(f"  ⚠ Moment calculation errors detected")

    # Generate plots
    print("\nGenerating plots...")
    plot_results(
        results_dict=results_dict,
        current_velocity=current_velocity,
        current_direction=current_direction,
        env_force_magnitude=env_force_magnitude,
        env_force_direction=env_force_direction,
        l_eff=l_eff,
        time_step=time_step,
        total_steps=total_steps
    )
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for scenario_name, (xs, ys, psis) in results_dict.items():
        # Final position
        final_x = xs[-1]
        final_y = ys[-1]
        distance = np.sqrt(final_x**2 + final_y**2)

        # Heading change
        psi_start = psis[0]
        psi_end = psis[-1]
        heading_change_deg = np.rad2deg(((psi_end - psi_start + np.pi) % (2*np.pi)) - np.pi)

        print(f"\n{SCENARIO_LABELS[scenario_name]}:")
        print(f"  Final position: ({final_x:.2f}, {final_y:.2f}) m")
        print(f"  Distance from start: {distance:.2f} m")
        print(f"  Heading change: {heading_change_deg:.1f}°")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
