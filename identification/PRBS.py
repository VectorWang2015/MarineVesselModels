import numpy as np
from scipy.signal import welch


def prbs_sequence(K, p_zero=0.0, rng=None):
    """
    generate a Kx1 sequence, each element with p_zero probability to be 0
        equal probability to be -1, +1 otherwise
    """
    if rng is None:
        rng = np.random.default_rng()
    seq = rng.choice([-1, 0, 1], size=K, p=[(1-p_zero)/2, p_zero, (1-p_zero)/2])
    return seq


def generate_dualthruster_prbs(
    T_total=600.0,        # total time in seconds
    dt=0.01,              # simulation time step
    Tc_sum=2.0,           # control interval in seconds: boost thrust nps
    Tc_diff=3.0,          # control interval in seconds: diff thrust nps
    N0=60.0,              # base control param
    As=40.0,              # boost control param (will be added to base)
    Ad=30.0,              # diff control param
    n_max=150.0,          # max control param, satuaration
    p_zero_sum=0.3,       # probability to generate a 0 order for boost nps
    p_zero_diff=0.2,      # probability to generate a 0 order for diff nps
    seed=None
):
    rng = np.random.default_rng(seed)
    # total time steps
    N = int(T_total / dt)

    # control steps number
    Ks = int(np.ceil(T_total / Tc_sum))
    Kd = int(np.ceil(T_total / Tc_diff))

    # generate control PRBS seqs
    s_ctrl = prbs_sequence(Ks, p_zero=p_zero_sum, rng=rng)  # in {-1,0,+1}
    d_ctrl = prbs_sequence(Kd, p_zero=p_zero_diff, rng=rng)

    # up-sampling to desired total time steps
    s_up = np.repeat(s_ctrl, repeats=int(Tc_sum/dt))[:N]
    d_up = np.repeat(d_ctrl, repeats=int(Tc_diff/dt))[:N]

    # scale, add
    n_sum   = N0 + As * s_up           # nΣ
    n_delta = Ad * d_up                 # nΔ

    n_L = n_sum - n_delta
    n_R = n_sum + n_delta

    # satuaration check
    n_L = np.clip(n_L, -n_max, n_max)
    n_R = np.clip(n_R, -n_max, n_max)

    # time line
    t = np.arange(N) * dt
    return t, n_L, n_R, n_sum, n_delta


def plot_PSD(ax, signal, dt):
    """
    plot power spectrual density for data
    """
    freqs, psd = welch(signal, 1/dt, window="hann", nperseg=4096)

    ax.semilogy(freqs, psd)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")

    return ax
