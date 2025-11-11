import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.Fossen import sample_hydro_params_2, sample_b_2, sample_thrust_2, Fossen
from MarineVesselModels.thrusters import NaiveDoubleThruster
from identification.least_square_methods import LeastSquareFossen, LeastSquareFossenSG
from identification.improved_ls import WeightedRidgeLSFossen
from identification.PRBS import generate_dualthruster_prbs
from demos.fossen_zigzag import fossen_zigzag
from demos.fossen_PRBS import fossen_prbs

from pickle import load


def dot_fn(signals, window_size=11, time_step=0.1):
    return savgol_filter(
        signals, window_length=window_size,
        polyorder=3, deriv=1, delta=time_step
    )

if __name__ == "__main__":
    train_data = np.load("./exp_data/Fossen_PRBS_0.1_900.npy")
    partial_data = np.load("./exp_data/partial_Fossen_PRBS_0.1_900.npy")
    noisy_train_data = np.load("./exp_data/noised_Fossen_PRBS_0.1_900.npy")
    with open("./exp_data/noised_Fossen_PRBS_0.1_900.pickle.obj", "rb") as fd:
        noisy_real_data = load(fd)
    noisy_partial_data = np.hstack(noisy_real_data["partials"])
    

    print(f"Sample size: {train_data.shape}")

    # identificate prbs train_data
    time_step = 0.1

    # train_data before this index, will be treated as init data
    # after this index, will be treated update data, which will be used for recursive update
    update_index = 1000
    
    identifier = WeightedRidgeLSFossen(time_step=time_step, weights=(1, 1, 2.5), lam=1e-3, enable_filter=True)
    result = identifier.identificate(
        us=train_data[3][:-1],
        vs=train_data[4][:-1],
        rs=train_data[5][:-1],
        dot_us=partial_data[3],
        dot_vs=partial_data[4],
        dot_rs=partial_data[5],
        taus=train_data[6:9, :-1],
    )
    print(f"Real params: \n{sample_hydro_params_2}")
    print(f"Identified result (PRBS): \n{result}")
    print(identifier.compute_residuals())
    print(20*"-")
    
    identifier = WeightedRidgeLSFossen(time_step=time_step, weights=(1, 1, 2.5), lam=1e-3, enable_filter=True)
    gradient_result = identifier.identificate(
        us=train_data[3],
        vs=train_data[4],
        rs=train_data[5],
        dot_us=dot_fn(train_data[3]),
        dot_vs=dot_fn(train_data[4]),
        dot_rs=dot_fn(train_data[5]),
        taus=train_data[6:9],
    )
    print(f"Real params: \n{sample_hydro_params_2}")
    print(f"Identified result (gradient PRBS): \n{gradient_result}")
    print(identifier.compute_residuals())
    print(20*"-")

    # train from noisy data
    identifier = WeightedRidgeLSFossen(time_step=time_step, weights=(1, 1, 2.5), lam=1e-3, enable_filter=True)
    noisy_result = identifier.identificate(
        us=noisy_train_data[3][:-1],
        vs=noisy_train_data[4][:-1],
        rs=noisy_train_data[5][:-1],
        dot_us=noisy_partial_data[3],
        dot_vs=noisy_partial_data[4],
        dot_rs=noisy_partial_data[5],
        taus=noisy_train_data[6:9, :-1],
    )
    print(f"Real params: \n{sample_hydro_params_2}")
    print(f"Identified result (noisy PRBS): \n{noisy_result}")
    print(identifier.compute_residuals())
    print(20*"-")

    identifier = WeightedRidgeLSFossen(time_step=time_step, weights=(1, 1, 2.5), lam=1e-3, enable_filter=True)
    noisy_gradient_result = identifier.identificate(
        us=noisy_train_data[3],
        vs=noisy_train_data[4],
        rs=noisy_train_data[5],
        dot_us=dot_fn(noisy_train_data[3]),
        dot_vs=dot_fn(noisy_train_data[4]),
        dot_rs=dot_fn(noisy_train_data[5]),
        taus=noisy_train_data[6:9],
    )
    print(f"Real params: \n{sample_hydro_params_2}")
    print(f"Identified result (noisy gradient PRBS): \n{noisy_gradient_result}")
    print(identifier.compute_residuals())
    print(20*"-")

    # identificate zigzag data
    zigzag_data = np.load("./exp_data/Fossen_zigzag_20_50_20_0.01.npy")
    time_step = 0.01

    identifier = LeastSquareFossen(time_step=time_step)
    zigzag_result = identifier.identificate(zigzag_data)
    print(f"Real params: \n{sample_hydro_params_2}")
    print(f"Identified result (zigzag): \n{zigzag_result}")
    print(20*"-")

    # use 1800 set as validation set
    dt = 0.1
    val_data = np.load("./exp_data/Fossen_PRBS_0.1_1800.npy")

    # check 90 seconds
    val_data = val_data[:,:900]
    taus = val_data[6:,:]
    print(val_data.shape)
    print(taus.shape)

    # generate data for PRBS params
    init_state = val_data[:6,0].reshape([6, 1])
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

    # generate data for dotV inferred PRBS params
    init_state = val_data[:6,0].reshape([6, 1])
    grad_prbs_states = init_state
    grad_prbs_simul = VesselSimulator(
        hydro_params=gradient_result,
        time_step=dt,
        model=Fossen,
        init_state=init_state
    )
    for i in range(taus.shape[1]):
        tau = taus[:, i].reshape([3, 1])
        cur_state = grad_prbs_simul.step(tau)
        grad_prbs_states = np.hstack((grad_prbs_states, cur_state))

    # generate data for dotV inferred PRBS params
    init_state = val_data[:6,0].reshape([6, 1])
    grad_prbs_states = init_state
    grad_prbs_simul = VesselSimulator(
        hydro_params=gradient_result,
        time_step=dt,
        model=Fossen,
        init_state=init_state
    )
    for i in range(taus.shape[1]):
        tau = taus[:, i].reshape([3, 1])
        cur_state = grad_prbs_simul.step(tau)
        grad_prbs_states = np.hstack((grad_prbs_states, cur_state))

    # generate data for PRBS params noisy model
    init_state = val_data[:6,0].reshape([6, 1])
    noised_prbs_states = init_state
    noised_prbs_simul = VesselSimulator(
        hydro_params=noisy_result,
        time_step=dt,
        model=Fossen,
        init_state=init_state
    )
    for i in range(taus.shape[1]):
        tau = taus[:, i].reshape([3, 1])
        cur_state = noised_prbs_simul.step(tau)
        noised_prbs_states = np.hstack((noised_prbs_states, cur_state))

    # generate data for inferred dot PRBS params noisy model
    init_state = val_data[:6,0].reshape([6, 1])
    noised_gradient_prbs_states = init_state
    noised_gradient_prbs_simul = VesselSimulator(
        hydro_params=noisy_gradient_result,
        time_step=dt,
        model=Fossen,
        init_state=init_state
    )
    for i in range(taus.shape[1]):
        tau = taus[:, i].reshape([3, 1])
        cur_state = noised_gradient_prbs_simul.step(tau)
        noised_gradient_prbs_states = np.hstack((noised_gradient_prbs_states, cur_state))

    # generate data for zigzag params
    init_state = val_data[:6,0].reshape([6, 1])
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
    ax.plot(zz_states[1], zz_states[0], label="ZigZag+LS")
    ax.plot(grad_prbs_states[1], grad_prbs_states[0], label="PRBS+ImprovedWLS+InferredDotV")
    ax.plot(prbs_states[1], prbs_states[0], label="PRBS+ImprovedWLS+RealDotV")
    ax.plot(noised_prbs_states[1], noised_prbs_states[0], label="noisy_model+PRBS+ImprovedWLS+RealDotV")
    ax.plot(noised_gradient_prbs_states[1], noised_gradient_prbs_states[0], label="noisy_model+PRBS+ImprovedWLS+InferredDotV")
    ax.legend()

    plt.show()