import numpy as np

from matplotlib import pyplot as plt
from json import loads


if __name__ == "__main__":
    time_step = 0.1
    record_name = "exp_data/noised_Fossen_PRBS_nps_0.1_900.json"
    with open(record_name) as fd:
        exp_record = loads(fd.read())

    xs = exp_record["xs"]
    ys = exp_record["ys"]
    psis = exp_record["psis"]
    us = exp_record["us"]
    vs = exp_record["vs"]
    rs = exp_record["rs"]
    Fxs = exp_record["Fxs"]
    Fys = exp_record["Fys"]
    Nrs = exp_record["Nrs"]
    nps_l = exp_record["rps_l"]
    nps_r = exp_record["rps_r"]

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.plot(ys, xs)
    plt.show()
    
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(np.arange(len(psis))*time_step, np.array(psis)/np.pi*180, label="$\psi$")
    axs[0].legend()
    axs[1].plot(np.arange(len(rs))*time_step, rs, label="$r$")
    axs[1].legend()
    axs[2].plot(np.arange(len(us))*time_step, us, label="$u$")
    axs[2].plot(np.arange(len(vs))*time_step, vs, label="$v$")
    axs[2].legend()
    axs[3].plot(np.arange(len(nps_l))*time_step, nps_l, label="left rps")
    axs[3].plot(np.arange(len(nps_r))*time_step, nps_r, label="right rps")
    axs[3].legend()
    plt.show()