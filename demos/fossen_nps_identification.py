import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.Fossen import sample_hydro_params_2, sample_b_2, sample_thrust_2, Fossen
from MarineVesselModels.thrusters import NpsDoubleThruster
from identification.least_square_methods import AlternatingLeastSquareFossen
from demos.fossen_zigzag import fossen_zigzag


if __name__ == "__main__":
    train_data = np.load("./demos/Fossen_nps_zigzag_20_100_20_0.01.npy")
    print(f"Sample size: {train_data.shape}")

    time_step = 0.01
    # train_data before this index, will be treated as init data
    # after this index, will be treated update data, which will be used for recursive update
    update_index = 100
    
    identifier = AlternatingLeastSquareFossen(time_step=time_step, b=sample_b_2)
    result = identifier.identificate(train_data=train_data)

    print(result)
    """
    identifier = RecursiveLeastSquareFossen(time_step=time_step)
    result = identifier.identificate(train_data=train_data[:, :update_index])
    print(f"Real params: \n{sample_hydro_params_2}")
    print(f"Identified result: \n{result}")
    
    thetas = []
    
    result = identifier.identificate(train_data=train_data[:, update_index:], thetas=thetas)
    print(f"Identified result: \n{result}")
    
    # thetas is now a 6xn array
    thetas = np.hstack(thetas)
    
    # plot src
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.plot(train_data[1, :], train_data[0, :], label="Train data")
    
    # plot prediction
    # init simulation
    time_step = 0.01
        
    init_psi = 0
    base_N = 50.0
    base_delta_N = 20.0
    zigzag_degrees = 20.0
        
    current_state = np.array([0, 0, init_psi, 0.659761, 0, 0]).reshape([6, 1])
        
    simulator = VesselSimulator(
        sample_hydro_params_2,
        time_step=time_step,
        model=Fossen,
        init_state=current_state,
    )
    thruster = NpsDoubleThruster(b=sample_b_2, **sample_thrust_params_2)
        
    states = fossen_zigzag(
        simulator=simulator,
        thruster=thruster,
        current_state=current_state,
        zigzag_degrees=zigzag_degrees,
        base_N=base_N,
        base_delta_N=base_delta_N,
    )
    
    ax.plot(states[1, :], states[0, :], label="Prediction")
    ax.legend()
    
    plt.show()
    
    # plot convergence
    fig, axs = plt.subplots(3, 2)
    
    def plot_convergence(ax, data, tag):
        ax.plot(np.arange(len(data)), data, label=tag)
        ax.set_xlabel("Iteration")
        ax.legend()
    
    plot_convergence(axs[0][0], thetas[0], "m11")
    plot_convergence(axs[1][0], thetas[1], "m22")
    plot_convergence(axs[2][0], thetas[2], "m33")
    plot_convergence(axs[0][1], thetas[3], "d11")
    plot_convergence(axs[1][1], thetas[4], "d22")
    plot_convergence(axs[2][1], thetas[5], "d33")
    plt.show()
    """