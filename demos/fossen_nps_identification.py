import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from json import loads

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.Fossen import (
    sample_hydro_params_2, sample_b_2, sample_thrust_2, sample_thrust_params_2,
)
from identification.improved_ls import AltLS

ref_params = {
    "m11": 50.05,
    "m22": 84.36,
    "m33": 17.21,
    "d11": 151.57,
    "d22": 132.5,
    "d33": 34.56,
    "c": -1.60e-4,
    "d": 5.04e-3,
}


def dot_fn(signals, window_size=11, time_step=0.1):
    return savgol_filter(
        signals, window_length=window_size,
        polyorder=3, deriv=1, delta=time_step
    )

if __name__ == "__main__":
    record_name = "exp_data/Fossen_PRBS_nps_0.1_900.json"
    with open(record_name) as fd:
        exp_record = loads(fd.read())

    xs = np.array(exp_record["xs"])
    ys = np.array(exp_record["ys"])
    psis = np.array(exp_record["psis"])
    us = np.array(exp_record["us"])
    vs = np.array(exp_record["vs"])
    rs = np.array(exp_record["rs"])
    dot_us = np.array(exp_record["dot_us"])
    dot_vs = np.array(exp_record["dot_vs"])
    dot_rs = np.array(exp_record["dot_rs"])
    Fxs = np.array(exp_record["Fxs"])
    Fys = np.array(exp_record["Fys"])
    Nrs = np.array(exp_record["Nrs"])
    rps_ls = np.array(exp_record["rps_l"])
    rps_rs = np.array(exp_record["rps_r"])
    F_x = np.array(exp_record["Fxs"])
    N_r = np.array(exp_record["Nrs"])

    # identificate prbs train_data
    time_step = 0.1

    # real value: c: -1.6e-4, d: 5e-3
    init_thru_params = {
        "c": -1e-4, "d": 3e-3,
    }
    hull_ls_params = {
        "weights":(1, 1, 2.5), "lam":1e-3, "enable_filter":True,
    }
    thru_ls_params = {
        "weights":(1, 1), "lam":1e-7, "enable_filter":True,
    }

    # train from noisy data
    identifier = AltLS(
        time_step=time_step, b=sample_b_2,
        init_thru_params=init_thru_params,
        hull_ls_params=hull_ls_params, thru_ls_params=thru_ls_params,
    )

    difference_records = []

    """
    for i in range(50):
        identified_result, relative_difference = identifier.identify_once(
            us=us,
            vs=vs,
            rs=rs,
            dot_us=dot_us,
            dot_vs=dot_vs,
            dot_rs=dot_rs,
            rps_ls=rps_ls,
            rps_rs=rps_rs,
        )
        print(f"Epoch: {i}")
        # print(identified_result)
        # print(relative_difference)

        if relative_difference is not None:
            difference_records.append(relative_difference)

    print("Real:")
    print(sample_hydro_params_2)
    print(sample_thrust_params_2)
    print("Identified:")
    print(identified_result)

    m11_diffs = [record["m11"] for record in difference_records]
    m33_diffs = [record["m33"] for record in difference_records]
    d11_diffs = [record["d11"] for record in difference_records]
    d33_diffs = [record["d33"] for record in difference_records]
    c_diffs = [record["c"] for record in difference_records]
    d_diffs = [record["d"] for record in difference_records]

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(np.arange(len(m11_diffs)), m11_diffs, label="m11")
    axs[0].plot(np.arange(len(d11_diffs)), d11_diffs, label="d11")
    axs[0].plot(np.arange(len(c_diffs)), c_diffs, label="c")
    axs[0].plot(np.arange(len(d_diffs)), d_diffs, label="d")
    axs[0].legend()
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Relative change")

    axs[1].plot(np.arange(len(m33_diffs)), m33_diffs, label="m11")
    axs[1].plot(np.arange(len(d33_diffs)), d33_diffs, label="d11")
    axs[1].plot(np.arange(len(c_diffs)), c_diffs, label="c")
    axs[1].plot(np.arange(len(d_diffs)), d_diffs, label="d")
    axs[1].legend()
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Relative change")

    axs[2].plot(np.arange(len(m33_diffs)), m33_diffs, label="m11")
    axs[2].plot(np.arange(len(m11_diffs)), d33_diffs, label="d11")
    axs[2].plot(np.arange(len(d_diffs)), d_diffs, label="d")
    axs[2].legend()
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Relative change")
    plt.show()
    """
    identified_result, relative_difference = identifier.identify(
        us=us,
        vs=vs,
        rs=rs,
        dot_us=dot_us,
        dot_vs=dot_vs,
        dot_rs=dot_rs,
        rps_ls=rps_ls,
        rps_rs=rps_rs,
        max_epochs=100,
        end_criteria=0.01,
    )
    print(f"Refs: {ref_params}")
    print(f"Identified: {identified_result}")
    print(f"Difference: {relative_difference}")

    ref_ratio = ref_params["m11"] / identified_result["m11"]
    # normalize identified_result
    for key in ref_params:
        normalized_data = identified_result[key] * ref_ratio
        error = ref_params[key] - normalized_data
        error_ratio = error / ref_params[key] * 100
        print(f"Noramlized {key}: {normalized_data}")
        print(f"Error {key}: {error_ratio}%")