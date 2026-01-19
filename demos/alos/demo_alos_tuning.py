import numpy as np
from matplotlib import pyplot as plt
import plot_utils

from MarineVesselModels.simulator import NoisyVesselSimulator, SimplifiedEnvironmentalDisturbanceSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen
from MarineVesselModels.noises import GaussMarkovNoiseGenerator

from control.pid import DoubleLoopHeadingPID, PIDAW
from control.los import AdaptiveLOSGuider


def test_alos(gamma, forward_dist, dt, control_every, total_exp_steps,
              waypoints, reached_threshold, env_force_magnitude, env_force_direction,
              desire_u, max_base_N, max_diff_N, time_step):
    """
    Run ALOS simulation with given gamma.

    Returns:
    --------
    dict with keys:
        xs, ys: position trajectories
        beta_hats: list of beta_hat values at each control step
        y_es: list of cross-track errors at each control step
        steps: number of simulation steps taken
        finished: bool whether reached final waypoint
    """
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
                dt=dt,
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
        dt=dt,
        u_min=-max_base_N,
        u_max=max_base_N,
    )

    guider = AdaptiveLOSGuider(
        waypoints=waypoints,
        reached_threshold=reached_threshold,
        forward_dist=forward_dist,
        dt=dt,
        gamma=gamma,
        beta_hat0=0.0,
        output_err_flag=False,
    )

    xs = []
    ys = []
    beta_hats = []
    y_es = []
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

        # record adaptation state (only at control steps to match beta_hat updates)
        if t % control_every == 0:
            beta_hats.append(guider.beta_hat)
            y_es.append(guider.calc_cross_track_error())

        xs.append(current_x)
        ys.append(current_y)

        # apply tau and step
        current_state = simulator.step(tau)

    return {
        'xs': xs,
        'ys': ys,
        'beta_hats': beta_hats,
        'y_es': y_es,
        'steps': len(xs),
        'finished': finished,
    }


if __name__ == "__main__":
    # Simulation parameters (same as comparison demo)
    time_step = 0.1
    total_exp_steps = 8000
    control_step = 0.2
    control_every = int(control_step / time_step)

    desire_u = 0.5
    max_base_N = 100.0
    max_diff_N = 60.0

    reached_threshold = 5.0
    forward_dist = 5.0

    # Environmental disturbance
    env_force_magnitude = 16.0
    env_force_direction = np.pi/4*3  # 135°

    # Waypoints: single segment
    waypoints = [(0, 0), (100, 100)]

    # Test different gamma values
    gamma_values = [0.0001, 0.0006, 0.001, 0.005, 0.01]

    results = {}
    print("Running ALOS tuning tests...")
    for gamma in gamma_values:
        print(f"  gamma = {gamma:.6f}", flush=True)
        results[gamma] = test_alos(
            gamma=gamma,
            forward_dist=forward_dist,
            dt=control_step,
            control_every=control_every,
            total_exp_steps=total_exp_steps,
            waypoints=waypoints,
            reached_threshold=reached_threshold,
            env_force_magnitude=env_force_magnitude,
            env_force_direction=env_force_direction,
            desire_u=desire_u,
            max_base_N=max_base_N,
            max_diff_N=max_diff_N,
            time_step=time_step,
        )
        print(f"    steps: {results[gamma]['steps']}, finished: {results[gamma]['finished']}")

    # Plot 1: Cross-track error over time (control steps)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    for gamma in gamma_values:
        y_es = results[gamma]['y_es']
        label = f'γ = {gamma:.6f}'
        t_control = np.arange(0, len(y_es)) * control_step
        ax1.plot(t_control, y_es, label=label, linewidth=1.5)

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Cross-track error y_e (m)')
    ax1.set_title('ALOS Cross-Track Error vs Time for Different γ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Beta_hat adaptation over time
    for gamma in gamma_values:
        beta_hats = results[gamma]['beta_hats']
        label = f'γ = {gamma:.6f}'
        t_control = np.arange(0, len(beta_hats)) * control_step
        ax2.plot(t_control, np.rad2deg(beta_hats), label=label, linewidth=1.5)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Estimated sideslip β_hat (deg)')
    ax2.set_title('ALOS Sideslip Adaptation for Different γ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Path trajectories
    for gamma in gamma_values:
        xs = results[gamma]['xs']
        ys = results[gamma]['ys']
        label = f'γ = {gamma:.6f}'
        ax3.plot(ys, xs, label=label, linewidth=1.5, alpha=0.8)

    # Add waypoints
    way_xs = [pt[0] for pt in waypoints]
    way_ys = [pt[1] for pt in waypoints]
    ax3.scatter(way_ys, way_xs, marker='x', color='black', s=100, label='Waypoints')
    ax3.plot(way_ys, way_xs, '--', color='black', alpha=0.5)

    # Add environmental force arrows
    plot_utils.add_force_direction_arrows(
        ax=ax3,
        direction_angle=env_force_direction,
        spacing=20.0,
    )

    ax3.set_xlabel('Y position (m)')
    ax3.set_ylabel('X position (m)')
    ax3.set_title('ALOS Path Following for Different γ')
    ax3.legend(loc='upper left')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance metrics bar chart
    rms_errors = []
    final_beta_hats = []
    for gamma in gamma_values:
        y_es = results[gamma]['y_es']
        rms = np.sqrt(np.mean(np.array(y_es)**2))
        rms_errors.append(rms)
        final_beta_hats.append(results[gamma]['beta_hats'][-1] if results[gamma]['beta_hats'] else 0)

    x_pos = np.arange(len(gamma_values))
    width = 0.35
    ax4.bar(x_pos - width/2, rms_errors, width, label='RMS y_e (m)', color='tab:blue')
    ax4_twin = ax4.twinx()
    ax4_twin.bar(x_pos + width/2, np.rad2deg(final_beta_hats), width, label='Final β_hat (deg)', color='tab:orange')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{g:.6f}' for g in gamma_values])
    ax4.set_xlabel('γ')
    ax4.set_ylabel('RMS cross-track error (m)')
    ax4_twin.set_ylabel('Final sideslip estimate (deg)')
    ax4.set_title('Performance Metrics vs γ')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('alos_tuning.png')
    print("Plot saved to alos_tuning.png")
    plt.close()

    # Print summary table
    print("\nALOS Tuning Summary:")
    print("γ".ljust(12) + "RMS y_e (m)".ljust(15) + "Final β_hat (deg)".ljust(20) + "Steps".ljust(10) + "Finished")
    print("-" * 70)
    for gamma in gamma_values:
        rms = np.sqrt(np.mean(np.array(results[gamma]['y_es'])**2))
        final_beta = results[gamma]['beta_hats'][-1] if results[gamma]['beta_hats'] else 0
        print(f"{gamma:.6f}".ljust(12) +
              f"{rms:.3f}".ljust(15) +
              f"{np.rad2deg(final_beta):.3f}".ljust(20) +
              f"{results[gamma]['steps']}".ljust(10) +
              f"{results[gamma]['finished']}")
