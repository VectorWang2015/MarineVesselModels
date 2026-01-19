import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt


state_data = np.load("./exp_data/Fossen_PRBS_0.1_900.npy")
partial_data = np.load("./exp_data/partial_Fossen_PRBS_0.1_900.npy")

partial_u = partial_data[3]
partial_v = partial_data[4]
partial_r = partial_data[5]

u = state_data[3]
v = state_data[4]
r = state_data[5]

calc_p_u = savgol_filter(
            u, window_length=11,
            polyorder=3, deriv=1, delta=0.1,
)[:-1]
calc_p_v = savgol_filter(
            v, window_length=11,
            polyorder=3, deriv=1, delta=0.1,
)[:-1]
calc_p_r = savgol_filter(
            r, window_length=11,
            polyorder=3, deriv=1, delta=0.1,
)[:-1]


fig, axs = plt.subplots(3, 1)


def compare_signal(
    ax,
    label,
    ref_sig,
    sig,
):
    residuals = ref_sig - sig
    t = np.arange(0, sig.shape[0])*0.1
    
    ax.set_ylabel(label)
    ax.plot(t, ref_sig, label="Reference")
    ax.plot(t, sig, label="Calculated")
    ax.plot(t, residuals, label="Residuals", color="red")
    ax.legend()
    
    print(np.average(ref_sig))
    print(np.average(residuals))

compare_signal(
    ax = axs[0],
    label = "u",
    ref_sig = partial_u,
    sig = calc_p_u,
)

compare_signal(
    ax = axs[1],
    label = "v",
    ref_sig = partial_v,
    sig = calc_p_v,
)
    
compare_signal(
    ax = axs[2],
    label = "r",
    ref_sig = partial_r,
    sig = calc_p_r,
)
    
plt.show()