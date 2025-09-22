import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.Fossen import sample_hydro_params_2, sample_b_2, sample_thrust_2, Fossen
from MarineVesselModels.thrusters import NaiveDoubleThruster
from identification.least_square_methods import LeastSquareFossen, LeastSquareFossenSG
from identification.PRBS import generate_dualthruster_prbs
from demos.fossen_zigzag import fossen_zigzag
from demos.fossen_PRBS import fossen_prbs


if __name__ == "__main__":
    train_data = np.load("./demos/Fossen_PRBS_0.1_1800.npy")
    print(f"Sample size: {train_data.shape}")

    # identificate prbs train_data
    time_step = 0.1

    # train_data before this index, will be treated as init data
    # after this index, will be treated update data, which will be used for recursive update
    update_index = 1000
    
    # identifier = RecursiveLeastSquareFossen(time_step=time_step)
    # result = identifier.identificate(train_data=train_data[:, :update_index])
    identifier = LeastSquareFossen(time_step=time_step)
    result = identifier.identificate(train_data=train_data)
    print(f"Real params: \n{sample_hydro_params_2}")
    print(f"Identified result (PRBS): \n{result}")
    
    # thetas = []
    
    # result = identifier.identificate(train_data=train_data[:, update_index:], thetas=thetas)
    # print(f"Identified result (full): \n{result}")
    
    # # thetas is now a 6xn array
    # thetas = np.hstack(thetas)
    
    # # plot convergence
    # fig, axs = plt.subplots(3, 2)
    
    # def plot_convergence(ax, data, tag):
    #     ax.plot(np.arange(len(data)), data, label=tag)
    #     ax.set_xlabel("Iteration")
    #     ax.legend()
    
    # plot_convergence(axs[0][0], thetas[0], "m11")
    # plot_convergence(axs[1][0], thetas[1], "m22")
    # plot_convergence(axs[2][0], thetas[2], "m33")
    # plot_convergence(axs[0][1], thetas[3], "d11")
    # plot_convergence(axs[1][1], thetas[4], "d22")
    # plot_convergence(axs[2][1], thetas[5], "d33")
    # plt.show()

    # identificate zigzag data
    zigzag_data = np.load("./demos/Fossen_zigzag_20_50_20_0.01.npy")
    time_step = 0.01

    identifier = LeastSquareFossenSG(time_step=time_step)
    zigzag_result = identifier.identificate(zigzag_data)
    print(f"Real params: \n{sample_hydro_params_2}")
    print(f"Identified result (zigzag): \n{zigzag_result}")

    # use 1800 set as validation set
    dt = 0.1
    val_data = np.load("./demos/Fossen_PRBS_0.1_1800.npy")

    # check 60 seconds
    val_data = val_data[:,:600]
    taus = val_data[6:,:]
    print(val_data.shape)
    print(taus.shape)

    # generate data for PRBS params
    init_state = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])
    prbs_states = init_state
    prbs_simul = VesselSimulator(
        hydro_params=result,
        time_step=dt,
        model=Fossen,
    )
    for i in range(taus.shape[1]):
        tau = taus[:, i].reshape([3, 1])
        cur_state = prbs_simul.step(tau)
        prbs_states = np.hstack((prbs_states, cur_state))

    # generate data for zigzag params
    init_state = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])
    zz_states = init_state
    zz_simul = VesselSimulator(
        hydro_params=zigzag_result,
        time_step=dt,
        model=Fossen,
    )
    for i in range(taus.shape[1]):
        tau = taus[:, i].reshape([3, 1])
        cur_state = zz_simul.step(tau)
        zz_states = np.hstack((zz_states, cur_state))

    # plot original
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.plot(val_data[1], val_data[0], label="Reference")
    ax.plot(prbs_states[1], prbs_states[0], label="PRBS")
    ax.plot(zz_states[1], zz_states[0], label="ZigZag")
    ax.legend()

    plt.show()

    # validation 2, neglect acceleration part
    dt = 0.1
    val_data = np.load("./demos/Fossen_PRBS_0.1_1800.npy")

    # check 60 seconds
    val_data = val_data[:,200:800]
    taus = val_data[6:,:]
    print(val_data.shape)
    print(taus.shape)

    # generate data for PRBS params
    init_state = val_data[:6,200].reshape([6, 1])
    prbs_states = init_state
    prbs_simul = VesselSimulator(
        hydro_params=result,
        time_step=dt,
        model=Fossen,
        init_state=init_state
    )
    for i in range(taus.shape[1]):
        tau = taus[:, i].reshape([3, 1])
        cur_state = prbs_simul.step(tau)
        prbs_states = np.hstack((prbs_states, cur_state))

    # generate data for zigzag params
    init_state = val_data[:6,200].reshape([6, 1])
    zz_states = init_state
    zz_simul = VesselSimulator(
        hydro_params=zigzag_result,
        time_step=dt,
        model=Fossen,
        init_state=init_state,
    )
    for i in range(taus.shape[1]):
        tau = taus[:, i].reshape([3, 1])
        cur_state = zz_simul.step(tau)
        zz_states = np.hstack((zz_states, cur_state))

    # plot original
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.plot(val_data[1], val_data[0], label="Reference")
    ax.plot(prbs_states[1], prbs_states[0], label="PRBS")
    ax.plot(zz_states[1], zz_states[0], label="ZigZag")
    ax.legend()

    plt.show()