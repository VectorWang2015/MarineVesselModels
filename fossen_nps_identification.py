import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from json import loads

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.Fossen import (
    sample_hydro_params_2, sample_b_2, sample_thrust_2, sample_thrust_params_2,
)
from identification.improved_ls import AltLS


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

    # real value: c: -1.6e-4, d: 5e-3
    init_thru_params = {
        "c": -1e-5, "d": 4e-3,
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

    for i in range(500):
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
        print(identified_result)
        print(relative_difference)