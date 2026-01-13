import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import VesselSimulator, SimplifiedEnvironmentalDisturbanceSimulator
from MarineVesselModels.Fossen import sample_hydro_params_2, sample_b_2, Fossen
from MarineVesselModels.thrusters import NaiveDoubleThruster

if __name__ == "__main__":
    # Simulation parameters
    time_step = 0.1
    total_steps = 3000
    env_force_magnitude = 3.0  # Newtons

    # Control input for circular movement: differential thrust
    # Left thrust negative, right thrust positive creates rotation
    left_thrust = 3  # N
    right_thrust = 10.0   # N

    thruster = NaiveDoubleThruster(b=sample_b_2)
    control_tau = thruster.newton_to_tau(l_thrust_N=left_thrust, r_thrust_N=right_thrust)

    # Create simulators
    simulators = {}

    # Baseline: no environmental disturbance
    simulators["No disturbance"] = VesselSimulator(
        hydro_params=sample_hydro_params_2,
        time_step=time_step,
        model=Fossen,
    )

    # Environmental disturbance at different directions
    directions = [0.0, np.pi/4, np.pi/2, np.pi]  # 0°, 45°, 90°, 180°
    direction_labels = ["0°", "45°", "90°", "180°"]

    for angle, label in zip(directions, direction_labels):
        simulators[f"Env force {label}"] = SimplifiedEnvironmentalDisturbanceSimulator(
            hydro_params=sample_hydro_params_2,
            time_step=time_step,
            env_force_magnitude=env_force_magnitude,
            env_force_direction=angle,
            model=Fossen,
        )

    # Run simulation
    trajectories = {}
    for name, sim in simulators.items():
        xs = []
        ys = []
        for _ in range(total_steps):
            state = sim.step(control_tau)
            xs.append(state[0][0])
            ys.append(state[1][0])
        trajectories[name] = (xs, ys)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")

    # Generate blue color palette for environmental disturbance trajectories
    n_env_trajectories = len(trajectories) - 1  # exclude "No disturbance"
    blues = plt.get_cmap('Blues')(np.linspace(0.3, 0.9, n_env_trajectories))
    
    for idx, (name, (xs, ys)) in enumerate(trajectories.items()):
        if name == "No disturbance":
            color = "black"
        else:
            color = blues[idx - 1]  # idx-1 because first is "No disturbance"
        ax.plot(ys, xs, label=name, color=color, linewidth=2)

    ax.set_xlabel("Y position (m)")
    ax.set_ylabel("X position (m)")
    ax.set_title(f"Circular Motion with Environmental Disturbance\n" +
                 f"Env force magnitude: {env_force_magnitude} N, " +
                 f"Control: left={left_thrust} N, right={right_thrust} N")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add starting point marker
    ax.scatter(0, 0, color="black", marker="o", s=100, label="Start", zorder=5)

    plt.tight_layout()
    plt.show()
