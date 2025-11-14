import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from json import loads

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.Fossen import (
    sample_hydro_params_2, sample_b_2, sample_thrust_2, sample_thrust_params_2,
)
from identification.improved_ls import WeightedRidgeLSThruster


def dot_fn(signals, window_size=11, time_step=0.1):
    return savgol_filter(
        signals, window_length=window_size,
        polyorder=3, deriv=1, delta=time_step
    )

if __name__ == "__main__":
    record_name = "exp_data/noised_Fossen_PRBS_nps_0.1_900.json"
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

    # train from noisy data
    identifier = WeightedRidgeLSThruster(
        time_step=time_step, weights=(1, 100), lam=1e-7, enable_filter=True, b=sample_b_2,
    )
    identifier.set_hydro_params(
        m11=sample_hydro_params_2["m11"],
        m22=sample_hydro_params_2["m22"],
        m33=sample_hydro_params_2["m33"],
        d11=sample_hydro_params_2["X_u"],
        d22=sample_hydro_params_2["Y_v"],
        d33=sample_hydro_params_2["N_r"],
    )
    noisy_result = identifier.identificate(
        us=us,
        vs=vs,
        rs=rs,
        dot_us=dot_us,
        dot_vs=dot_vs,
        dot_rs=dot_rs,
        rps_ls=rps_ls,
        rps_rs=rps_rs,
    )
    print(f"Real params: \n{sample_thrust_params_2}")
    print(f"Identified result (noisy PRBS): \n{noisy_result}")
    print(f"Dataset average Fx(input): {np.mean(F_x)}, average Nr(input): {np.mean(N_r)}")
    print(identifier.compute_residuals())
    print(20*"-")

    calcul_Fx = identifier.y[0::2].flatten()
    calcul_Nr = identifier.y[1::2].flatten()

    res_Fx = F_x[0:9000] - calcul_Fx
    res_Nr = N_r[0:9000] - calcul_Nr

    plt.plot(np.arange(len(res_Fx)), res_Fx)
    plt.plot(np.arange(len(res_Nr)), res_Nr)
    plt.show()