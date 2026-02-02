import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import SimplifiedEnvironmentalDisturbanceSimulator
from MarineVesselModels.Fossen import sample_hydro_params_2, sample_b_2, FossenWithCurrent
from MarineVesselModels.thrusters import NaiveDoubleThruster
from plot_utils import add_force_direction_arrows


"""
Current rotation test with ocean current validation.

This demo tests the FossenWithCurrent model by simulating vessel motion under
different current directions with asymmetric thrust configurations.

Features:
1. Three thrust configurations: (25N,75N), (75N,25N), and (0N,0N) for drift validation
2. Three current directions: 0° (North), 45° (NE), 90° (East)
3. Drift validation compares theoretical vs simulated drift for zero-thrust scenario
4. Two plotting modes:
   - PNG output: Combined 2x2 figure with all scenarios
   - Interactive display: Separate figures for each scenario

The zero-thrust drift test validates that the vessel drifts with the current,
though simulated drift will be less than theoretical due to acceleration time.
"""


def run_rotation_test(
        current_direction,
        current_velocity,
        force_direction,
        force_magnitude,
        l_eff,
        control_tau,
        time_step,
        total_steps
):
    """
    Run rotation simulation with given current parameters.

    Returns:
        tuple: (xs, ys) trajectory coordinates
    """
    simulator = SimplifiedEnvironmentalDisturbanceSimulator(
        hydro_params=sample_hydro_params_2,
        time_step=time_step,
        model=FossenWithCurrent,
        env_force_magnitude=force_magnitude,  # No environmental force, only current
        env_force_direction=force_direction,
        l_eff=l_eff,
        current_velocity=current_velocity,
        current_direction=current_direction,
    )

    xs, ys = [], []
    for _ in range(total_steps):
        state = simulator.step(control_tau)
        xs.append(state[0][0])
        ys.append(state[1][0])

    return xs, ys


def plot_scenario(ax, label, direction, baseline_xs, baseline_ys,
                  xs1, ys1, xs2, ys2, xs3, ys3, current_velocity):
    """
    Plot a single current scenario on given axis.

    Args:
        ax: matplotlib axis object
        label: scenario label string
        direction: current direction in radians (NED)
        baseline_xs, baseline_ys: baseline trajectory (no current)
        xs1, ys1: config 1 trajectory (25N, 75N)
        xs2, ys2: config 2 trajectory (75N, 25N)
        xs3, ys3: config 3 trajectory (0N, 0N)
        current_velocity: current velocity in m/s
    """
    # Plot baseline trajectory (no current)
    ax.plot(baseline_ys, baseline_xs, color='black', linestyle='--',
            linewidth=1.5, alpha=0.7, label='No current (baseline)')

    # Plot trajectory with current - config 1 (25N, 75N) - blue line
    ax.plot(ys1, xs1, color='blue', linewidth=1.5, alpha=0.8, label='25N, 75N')

    # Plot trajectory with current - config 2 (75N, 25N) - red line
    ax.plot(ys2, xs2, color='red', linewidth=1.5, alpha=0.5, label='75N, 25N')

    # Plot trajectory with current - config 3 (0N, 0N) - green line for drift validation
    ax.plot(ys3, xs3, color='green', linewidth=1.5, alpha=0.7, label='0N, 0N (drift)')

    # Add force direction arrows in background
    add_force_direction_arrows(
        ax=ax,
        direction_angle=direction,
        spacing=1.0,
        color='0.2',
        alpha=0.2,
        coord_system='NED'
    )

    ax.set_aspect('equal')
    ax.set_xlabel('Y position (East) [m]')
    ax.set_ylabel('X position (North) [m]')
    ax.set_title(f'Current: {label}\nVelocity: {current_velocity} m/s')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # Add text with drift information
    # Config 1 drift relative to baseline (with thrust)
    drift_x1 = xs1[-1] - baseline_xs[-1]
    drift_y1 = ys1[-1] - baseline_ys[-1]
    drift_distance1 = np.sqrt(drift_x1**2 + drift_y1**2)

    # Config 3 drift from start (zero thrust)
    drift_distance3 = np.sqrt(xs3[-1]**2 + ys3[-1]**2)

    drift_text = f'Drift (25,75): {drift_distance1:.2f} m\nDrift (0,0): {drift_distance3:.2f} m'
    ax.text(0.05, 0.95, drift_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=8)


def plot_summary(ax, baseline_xs, baseline_ys, results_config1,
                 direction_labels, current_velocity):
    """
    Plot summary of all current directions for config 1.
    """
    # Plot baseline
    ax.plot(baseline_ys, baseline_xs, color='black', linestyle='--',
           linewidth=2, alpha=0.7, label='No current (baseline)')

    # Plot all current scenarios for config 1 (25N, 75N) only
    for label in direction_labels:
        xs1, ys1, direction = results_config1[label]
        ax.plot(ys1, xs1, linewidth=1.5, alpha=0.8, label=f'{label}')

    ax.set_aspect('equal')
    ax.set_xlabel('Y position (East) [m]')
    ax.set_ylabel('X position (North) [m]')
    ax.set_title(f'Summary: All Current Directions\n'
                f'Control: left=25N, right=75N\n'
                f'Current velocity: {current_velocity} m/s')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)


if __name__ == "__main__":
    # Simulation parameters
    time_step = 0.1
    total_steps = 1000  # 100 seconds (reduced from 200)

    # Current settings
    current_velocity = 0.1  # m/s
    force_magnitude = 5  # N
    l_eff = 0.1 # m
    current_directions = [0.0, np.pi/4, np.pi/2]  # 0°, 45°, 90° in NED coordinates
    direction_labels = ["0° (North)", "45° (NE)", "90° (East)"]

    # Control input configurations
    # Configuration 1: left 25N, right 75N creates forward + rotational motion
    left_thrust_1 = 25.0  # N
    right_thrust_1 = 75.0  # N

    # Configuration 2: left 75N, right 25N for symmetry check
    left_thrust_2 = 75.0  # N
    right_thrust_2 = 25.0  # N

    # Configuration 3: zero thrust for drift validation test
    left_thrust_3 = 0.0  # N
    right_thrust_3 = 0.0  # N

    thruster = NaiveDoubleThruster(b=sample_b_2)
    control_tau_1 = thruster.newton_to_tau(l_thrust_N=left_thrust_1, r_thrust_N=right_thrust_1)
    control_tau_2 = thruster.newton_to_tau(l_thrust_N=left_thrust_2, r_thrust_N=right_thrust_2)
    control_tau_3 = thruster.newton_to_tau(l_thrust_N=left_thrust_3, r_thrust_N=right_thrust_3)

    # Run baseline simulation (no current) for configuration 1
    print("Running baseline simulation (no current, config 1)...")
    baseline_xs, baseline_ys = run_rotation_test(
        current_direction=0,
        current_velocity=0,
        force_direction=0,
        force_magnitude=0,
        l_eff=l_eff,
        control_tau=control_tau_1,
        time_step=time_step,
        total_steps=total_steps,
    )

    # Run simulations with different current directions for all configurations
    results_config1 = {}
    results_config2 = {}
    results_config3 = {}
    for direction, label in zip(current_directions, direction_labels):
        print(f"Running simulation with current direction {label} (config 1: 25N, 75N)...")
        xs1, ys1 = run_rotation_test(
            current_direction=direction,
            current_velocity=current_velocity,
            force_direction=direction,
            force_magnitude=force_magnitude,
            l_eff=l_eff,
            control_tau=control_tau_1,
            time_step=time_step,
            total_steps=total_steps,
        )
        results_config1[label] = (xs1, ys1, direction)

        print(f"Running simulation with current direction {label} (config 2: 75N, 25N)...")
        xs2, ys2 = run_rotation_test(
            current_direction=direction,
            current_velocity=current_velocity,
            force_direction=direction,
            force_magnitude=force_magnitude,
            l_eff=l_eff,
            control_tau=control_tau_2,
            time_step=time_step,
            total_steps=total_steps,
        )
        results_config2[label] = (xs2, ys2, direction)

        print(f"Running simulation with current direction {label} (config 3: 0N, 0N)...")
        xs3, ys3 = run_rotation_test(
            current_direction=direction,
            current_velocity=current_velocity,
            force_direction=direction,
            force_magnitude=force_magnitude,
            l_eff=l_eff,
            control_tau=control_tau_3,
            time_step=time_step,
            total_steps=total_steps,
        )
        results_config3[label] = (xs3, ys3, direction)

    # Drift validation analysis for zero-thrust scenario
    print("\n" + "="*60)
    print("DRIFT VALIDATION ANALYSIS (Zero-thrust scenario)")
    print("="*60)
    total_time = total_steps * time_step
    print(f"Total simulation time: {total_time:.1f} s")
    print(f"Current velocity: {current_velocity} m/s")
    print("\nComparison of theoretical vs simulated drift:")
    print("-"*60)
    print(f"{'Current':<20} {'Theoretical drift [m]':<25} {'Simulated drift [m]':<25} {'Error [m]':<15}")
    print("-"*60)

    for label in direction_labels:
        xs3, ys3, direction = results_config3[label]
        # Theoretical drift if vessel instantly reaches current velocity
        theoretical_drift_x = current_velocity * total_time * np.cos(direction)
        theoretical_drift_y = current_velocity * total_time * np.sin(direction)
        theoretical_distance = np.sqrt(theoretical_drift_x**2 + theoretical_drift_y**2)

        # Simulated drift (vessel starts at 0,0)
        simulated_drift_x = xs3[-1] - xs3[0]  # xs3[0] should be 0
        simulated_drift_y = ys3[-1] - ys3[0]  # ys3[0] should be 0
        simulated_distance = np.sqrt(simulated_drift_x**2 + simulated_drift_y**2)

        # Error
        error_distance = abs(theoretical_distance - simulated_distance)

        print(f"{label:<20} {theoretical_distance:<25.2f} {simulated_distance:<25.2f} {error_distance:<15.2f}")

    print("="*60)
    print("Note: Theoretical drift assumes vessel instantly reaches current velocity.")
    print("Simulated drift will be less due to acceleration time.")

    # ==== PNG OUTPUT: Combined 2x2 figure ====
    print("\nCreating combined figure for PNG output (2x2 layout)...")
    fig_combined, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Plot individual scenarios (first 3 subplots)
    for idx, label in enumerate(direction_labels):
        ax = axes[idx]
        xs1, ys1, direction = results_config1[label]
        xs2, ys2, _ = results_config2[label]
        xs3, ys3, _ = results_config3[label]

        # Use the plotting function (config2 plotted only for idx==0 to match original)
        plot_config2 = (idx == 0)  # Only plot config2 for first scenario (0° North)

        # Plot baseline trajectory (no current)
        ax.plot(baseline_ys, baseline_xs, color='black', linestyle='--',
                linewidth=1.5, alpha=0.7, label='No current (baseline)')

        # Plot trajectory with current - config 1 (25N, 75N) - blue line
        ax.plot(ys1, xs1, color='blue', linewidth=1.5, alpha=0.8, label='25N, 75N')

        # Plot trajectory with current - config 2 (75N, 25N) - red line only for 0° current direction
        if plot_config2:
            ax.plot(ys2, xs2, color='red', linewidth=1.5, alpha=0.5, label='75N, 25N')

        # Plot trajectory with current - config 3 (0N, 0N) - green line for drift validation
        ax.plot(ys3, xs3, color='green', linewidth=1.5, alpha=0.7, label='0N, 0N (drift)')

        # Add force direction arrows in background
        add_force_direction_arrows(
            ax=ax,
            direction_angle=direction,
            spacing=1.0,
            color='0.2',
            alpha=0.2,
            coord_system='NED'
        )

        ax.set_aspect('equal')
        ax.set_xlabel('Y position (East) [m]')
        ax.set_ylabel('X position (North) [m]')
        ax.set_title(f'Current: {label}\nVelocity: {current_velocity} m/s')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

        # Add text with drift information
        # Config 1 drift relative to baseline (with thrust)
        drift_x1 = xs1[-1] - baseline_xs[-1]
        drift_y1 = ys1[-1] - baseline_ys[-1]
        drift_distance1 = np.sqrt(drift_x1**2 + drift_y1**2)

        # Config 3 drift from start (zero thrust)
        drift_distance3 = np.sqrt(xs3[-1]**2 + ys3[-1]**2)

        drift_text = f'Drift (25,75): {drift_distance1:.2f} m\nDrift (0,0): {drift_distance3:.2f} m'
        ax.text(0.05, 0.95, drift_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=8)

    # Plot summary in the last subplot
    ax_summary = axes[3]
    plot_summary(ax_summary, baseline_xs, baseline_ys, results_config1,
                 direction_labels, current_velocity)

    plt.tight_layout()

    # Save combined figure
    output_path = "current_rotation_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Combined figure saved to: {output_path}")

    # Close combined figure to avoid showing it
    plt.close(fig_combined)

    # ==== INTERACTIVE DISPLAY: Separate figures ====
    print("\nCreating separate figures for interactive display...")

    # Create separate figures for each scenario
    for idx, label in enumerate(direction_labels):
        fig_separate = plt.figure(figsize=(10, 8))
        ax = fig_separate.add_subplot(111)
        xs1, ys1, direction = results_config1[label]
        xs2, ys2, _ = results_config2[label]
        xs3, ys3, _ = results_config3[label]

        # Use the plotting function
        plot_scenario(ax, label, direction, baseline_xs, baseline_ys,
                     xs1, ys1, xs2, ys2, xs3, ys3, current_velocity)

        # Adjust legend for single figure
        ax.legend(loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.show(block=False)  # Show without blocking

    # Create separate figure for summary
    fig_summary = plt.figure(figsize=(10, 8))
    ax_summary = fig_summary.add_subplot(111)
    plot_summary(ax_summary, baseline_xs, baseline_ys, results_config1,
                 direction_labels, current_velocity)
    plt.tight_layout()
    plt.show(block=False)

    print("\nAll figures displayed. Close windows to exit.")
    plt.show()  # Keep all figures open
