import numpy as np
from matplotlib import pyplot as plt
import plot_utils

from MarineVesselModels.simulator import NoisyVesselSimulator, SimplifiedEnvironmentalDisturbanceSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen
from MarineVesselModels.noises import GaussMarkovNoiseGenerator

from control.pid import DoubleLoopHeadingPID, PIDAW
from control.los import DynamicDistLOSGuider, AdaptiveLOSGuider


def test_guider_with_history(simulator, thruster, diff_controller, u_controller, guider,
                             current_state, total_exp_steps, control_every, desire_u):
    """
    Returns trajectory and adaptation history.
    """
    xs = []
    ys = []
    beta_hats = []
    y_es = []
    segment_indices = []
    finished = False

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
                finished = True
                break
            # calculate new tau
            diff_control_signal, ref_r = diff_controller.step(psi_ref=desired_psi, psi=current_psi, r=current_r)
            velo_control_signal = u_controller.step(sp=desire_u, y=current_u)

            psi_err = ((desired_psi - current_psi) + np.pi) % (2*np.pi) - np.pi

            left = velo_control_signal + diff_control_signal
            right = velo_control_signal - diff_control_signal
            tau = thruster.newton_to_tau(left, right)

        # record adaptation state (only at control steps)
        if t % control_every == 0:
            if hasattr(guider, 'beta_hat'):
                beta_hats.append(guider.beta_hat)
            else:
                beta_hats.append(0.0)
            if hasattr(guider, 'calc_cross_track_error'):
                y_es.append(guider.calc_cross_track_error())
            else:
                y_es.append(0.0)
            # track segment index (crude: count of remaining waypoints)
            segment_indices.append(len(guider.reference_path) if hasattr(guider, 'reference_path') else 0)

        xs.append(current_x)
        ys.append(current_y)

        # apply tau and step
        current_state = simulator.step(tau)

    return {
        'xs': xs,
        'ys': ys,
        'beta_hats': beta_hats,
        'y_es': y_es,
        'segment_indices': segment_indices,
        'steps': len(xs),
        'finished': finished,
    }


if __name__ == "__main__":
    # Simulation parameters
    time_step = 0.1
    total_exp_steps = 12000  # Longer for square path
    control_step = 0.2
    control_every = int(control_step / time_step)

    desire_u = 0.5
    max_base_N = 100.0
    max_diff_N = 60.0

    reached_threshold = 5.0
    forward_dist = 5.0

    # Environmental disturbance (constant)
    env_force_magnitude = 16.0
    env_force_direction = np.pi/4*3  # 135°

    # Square path: (0,0) -> (100,0) -> (100,100) -> (0,100) -> (0,0)
    waypoints = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]

    # Initialize figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Colors
    colors = {'Dynamic LOS': 'tab:blue', 'Adaptive LOS': 'tab:orange', 'Adaptive LOS (reset)': 'tab:green'}

    # Test both guiders
    guider_configs = [
        ('Dynamic LOS', DynamicDistLOSGuider, {'forward_dist': forward_dist, 'output_err_flag': False}),
        ('Adaptive LOS', AdaptiveLOSGuider, {'forward_dist': forward_dist, 'gamma': 0.005, 'beta_hat0': 0.0,
                                             'dt': control_step, 'output_err_flag': False}),
        ('Adaptive LOS (reset)', AdaptiveLOSGuider, {'forward_dist': forward_dist, 'gamma': 0.005, 'beta_hat0': 0.0,
                                                     'dt': control_step, 'output_err_flag': False,
                                                     'reset_beta_on_segment_change': True}),
    ]

    results = {}

    print("Running multi-segment path following tests...")
    for guider_name, GuiderClass, guider_kwargs in guider_configs:
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
        results[guider_name] = test_guider_with_history(
            simulator, thruster, diff_controller, u_controller, guider,
            current_state, total_exp_steps, control_every, desire_u)

        print(f"  {guider_name}: {results[guider_name]['steps']} steps, finished: {results[guider_name]['finished']}")

    # Plot 1: Path trajectories
    for guider_name in results:
        xs = results[guider_name]['xs']
        ys = results[guider_name]['ys']
        ax1.plot(ys, xs, label=guider_name, color=colors[guider_name], linewidth=2, alpha=0.8)

    # Add waypoints
    way_xs = [pt[0] for pt in waypoints]
    way_ys = [pt[1] for pt in waypoints]
    ax1.scatter(way_ys, way_xs, marker='x', color='black', s=100, label='Waypoints')
    ax1.plot(way_ys, way_xs, '--', color='black', alpha=0.5)

    # Add environmental force arrows
    plot_utils.add_force_direction_arrows(
        ax=ax1,
        direction_angle=env_force_direction,
        spacing=20.0,
    )

    ax1.set_xlabel('Y position (m)')
    ax1.set_ylabel('X position (m)')
    ax1.set_title('Multi-Segment Path Following')
    ax1.legend(loc='upper left')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cross-track error over time (control steps)
    for guider_name in results:
        y_es = results[guider_name]['y_es']
        t_control = np.arange(0, len(y_es)) * control_step
        ax2.plot(t_control, y_es, label=guider_name, color=colors[guider_name], linewidth=1.5)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Cross-track error y_e (m)')
    ax2.set_title('Cross-Track Error vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Beta_hat adaptation (for all Adaptive LOS variants)
    for guider_name in results:
        if 'Adaptive LOS' in guider_name:
            beta_hats = results[guider_name]['beta_hats']
            t_control = np.arange(0, len(beta_hats)) * control_step
            ax3.plot(t_control, np.rad2deg(beta_hats), label=guider_name, color=colors[guider_name], linewidth=1.5)

    if any('Adaptive LOS' in guider_name for guider_name in results):
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Estimated sideslip β_hat (deg)')
        ax3.set_title('ALOS Sideslip Adaptation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Segment transitions (vertical lines) with beta_hat for both variants
    adaptive_keys = [guider_name for guider_name in results if 'Adaptive LOS' in guider_name]
    if adaptive_keys:
        # Use segment indices from first adaptive LOS (same for all)
        segment_indices = results[adaptive_keys[0]]['segment_indices']
        t_control = np.arange(0, len(segment_indices)) * control_step

        # Plot vertical lines at segment transitions
        changes = []
        for i in range(1, len(segment_indices)):
            if segment_indices[i] != segment_indices[i-1]:
                changes.append(i)

        for idx in changes:
            ax4.axvline(t_control[idx], color='red', linestyle='--', alpha=0.5, linewidth=1, label='Segment change' if idx == changes[0] else '')

        # Plot beta_hat for each adaptive variant
        for guider_name in adaptive_keys:
            beta_hats = results[guider_name]['beta_hats']
            ax4.plot(t_control[:len(beta_hats)], np.rad2deg(beta_hats), label=guider_name, color=colors[guider_name], linewidth=1.5)

        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('β_hat (deg)')
        ax4.set_title('ALOS Adaptation with Segment Transitions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Plot 5: Heading error comparison
    # (Not recorded in current function, skip for now)
    #ax5.axis('off')

    # Plot 6: RMS error per segment (placeholder)
    #ax6.axis('off')

    plt.tight_layout()
    plt.savefig('alos_multisegment.png')
    print("Plot saved to alos_multisegment.png")
    plt.close()

    # Print summary
    print("\nMulti-Segment Test Summary:")
    print("Guider".ljust(20) + "Steps".ljust(10) + "Finished".ljust(10) + "Max |y_e|".ljust(12) + "RMS y_e".ljust(12))
    print("-" * 70)
    for guider_name in results:
        y_es = results[guider_name]['y_es']
        max_ye = np.max(np.abs(y_es))
        rms_ye = np.sqrt(np.mean(np.array(y_es)**2))
        print(f"{guider_name.ljust(20)}"
              f"{results[guider_name]['steps']}".ljust(10) +
              f"{results[guider_name]['finished']}".ljust(10) +
              f"{max_ye:.3f}".ljust(12) +
              f"{rms_ye:.3f}".ljust(12))

    # Print beta_hat summary for each Adaptive LOS variant
    adaptive_keys = [guider_name for guider_name in results if 'Adaptive LOS' in guider_name]
    for guider_name in adaptive_keys:
        beta_hats = results[guider_name]['beta_hats']
        print(f"\n{guider_name} Final β_hat: {np.rad2deg(beta_hats[-1]):.3f} deg")
        print(f"{guider_name} β_hat range: [{np.rad2deg(np.min(beta_hats)):.3f}, {np.rad2deg(np.max(beta_hats)):.3f}] deg")
