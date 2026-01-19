import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from json import loads

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.Fossen import (
    sample_hydro_params_2, sample_b_2, sample_thrust_2, sample_thrust_params_2,
    Fossen,
)
from MarineVesselModels.thrusters import NpsDoubleThruster
from identification.improved_ls import AltLS, WeightedRidgeLSFossenRps

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

def plot_fossen_nps(
        ax,
        label,
        hydro_params,
        l_seq,
        r_seq,
        current_state = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1]),
):
    xs = []
    ys = []

    hydro_keys = ["m11", "m22", "m33", "d11", "d22", "d33"]
    thrus_keys = ["c", "d"]
    
    simulator = VesselSimulator(
        hydro_params={k: hydro_params[k] for k in hydro_keys},
        time_step=time_step,
        model=Fossen,
        init_state=current_state,
        output_partial=True,
    )
    thruster = NpsDoubleThruster(b=sample_b_2, **{k: hydro_params[k] for k in thrus_keys})

    for rps_l, rps_r in zip(l_seq, r_seq):
        # use rps and current u as input, to compute tau
        u = current_state[3][0]
        x, y = current_state[0][0], current_state[1][0]
        xs.append(x)
        ys.append(y)

        tau = thruster.n_to_tau(l_nps=rps_l, r_nps=rps_r, u=u)
        current_state, current_partial_state = simulator.step(tau)

    ax.plot(ys, xs, label=label)


def dot_fn(signals, window_size=11, time_step=0.1):
    return savgol_filter(
        signals, window_length=window_size,
        polyorder=3, deriv=1, delta=time_step
    )
    

if __name__ == "__main__":
    record_name = "exp_data/noised_Fossen_PRBS_nps_0.1_900.json"
    with open(record_name) as fd:
        exp_record = loads(fd.read())

    # original state [x,y,p,u,v,r] contains an extra record
    xs = np.array(exp_record["xs"])[:-1]
    ys = np.array(exp_record["ys"])[:-1]
    psis = np.array(exp_record["psis"])[:-1]
    us = np.array(exp_record["us"])[:-1]
    vs = np.array(exp_record["vs"])[:-1]
    rs = np.array(exp_record["rs"])[:-1]
    dot_us = np.array(exp_record["dot_us"])
    dot_vs = np.array(exp_record["dot_vs"])
    dot_rs = np.array(exp_record["dot_rs"])
    Fxs = np.array(exp_record["Fxs"])
    Fys = np.array(exp_record["Fys"])
    Nrs = np.array(exp_record["Nrs"])
    rps_ls = np.array(exp_record["rps_l"])
    rps_rs = np.array(exp_record["rps_r"])
    
    print(ref_params)

    # identify using ALS
    # identify prbs train_data
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

    ref_ratio = ref_params["m11"] / identified_result["m11"]

    ALS_noised_unaligned = {k: float(identified_result[k]) for k in identified_result}
    ALS_noised = {k: float(identified_result[k]*ref_ratio) for k in identified_result}

    print("-"*20)
    print("Results identified using ALS, from noised data:")
    print(ALS_noised_unaligned)
    print("Aligned")
    print(ALS_noised)

    """
    # test ALS step by step and trace identified results
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
    time_step = 0.1

    identifier = WeightedRidgeLSFossenRps(
        time_step=time_step, m11=ref_params["m11"] , b=sample_b_2,
        SG_window=11, weights=(1, 1, 1),
        lam=0.001, enable_filter=True,
    )

    m11_noised = identifier.identify(
        us=us,
        vs=vs,
        rs=rs,
        dot_us=dot_us,
        dot_vs=dot_vs,
        dot_rs=dot_rs,
        rps_ls=rps_ls,
        rps_rs=rps_rs,
    )

    print("-"*20)
    print("Results identified using m11LS, from noised data:")
    print(m11_noised)

    # reading validation data
    record_name = "exp_data/Fossen_PRBS_nps_0.1_90.json"
    with open(record_name) as fd:
        exp_record = loads(fd.read())
    xs = np.array(exp_record["xs"])[:-1]
    ys = np.array(exp_record["ys"])[:-1]
    psis = np.array(exp_record["psis"])[:-1]
    us = np.array(exp_record["us"])[:-1]
    vs = np.array(exp_record["vs"])[:-1]
    rs = np.array(exp_record["rs"])[:-1]
    dot_us = np.array(exp_record["dot_us"])
    dot_vs = np.array(exp_record["dot_vs"])
    dot_rs = np.array(exp_record["dot_rs"])
    Fxs = np.array(exp_record["Fxs"])
    Fys = np.array(exp_record["Fys"])
    Nrs = np.array(exp_record["Nrs"])
    rps_ls = np.array(exp_record["rps_l"])
    rps_rs = np.array(exp_record["rps_r"])

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.plot(ys, xs, "--", color="k", alpha=0.8, label="Reference")

    plot_fossen_nps(
        ax=ax, label="m11 LS method, noised data",
        hydro_params=m11_noised, l_seq=rps_ls, r_seq=rps_rs,
    )
    plot_fossen_nps(
        ax=ax, label="ALS method, noised data",
        hydro_params=ALS_noised_unaligned, l_seq=rps_ls, r_seq=rps_rs,
    )
    """
    plot_fossen_nps(
        ax=ax, label="ALS method, noised data, params aligned m11",
        hydro_params=ALS_noised, l_seq=rps_ls, r_seq=rps_rs,
    )
    """
    
    ax.legend()
    plt.show()