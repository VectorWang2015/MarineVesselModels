import numpy as np
from typing import Optional, Union


class GaussMarkovNoiseGenerator():
    def __init__(
            self,
            dt: float, tau: Union[float, np.ndarray], Sigma: np.ndarray,
            x0: Optional[np.ndarray] = None, rng: Optional[np.random.Generator] = None,
            jitter: float = 0.0, from_stationary: bool = True,
    ):
        if rng is None:
            rng = np.random.default_rng()
        k = Sigma.shape[0]
        assert Sigma.shape == (k, k)

        self.rng = rng
        self.k = k
    
        if type(tau) is not np.ndarray:
            tau = np.full(k, tau)
        
        self.tau = tau
    
        phi = np.exp(-dt / tau)
        Phi = np.diag(phi)
        Qd = Sigma - (Phi @ Sigma @ Phi.T)
        if jitter > 0.0:
            Qd = Qd + jitter * np.eye(k)
        L = np.linalg.cholesky(Qd)

        self.Phi = Phi
        self.Qd = Qd
        self.L = L
    
        if x0 is None:
            if from_stationary:
                self.x = rng.multivariate_normal(np.zeros(k), Sigma)
            else:
                self.x = np.zeros(k)
        else:
            self.x = x0
    
    def step(self):
        # equals: rng.multivariate_normal(np.zeros(k), Qd)
        # pre-cholesky saves computation
        self.x = self.Phi @ self.x + self.L @ self.rng.standard_normal(self.k)
        return self.x


class GaussianNoiseGenerator:
    def __init__(
            self,
            Sigma: np.ndarray,
            rng: Optional[np.random.Generator] = None,
            jitter: float = 0.0,
    ):
        if rng is None:
            rng = np.random.default_rng()

        k = Sigma.shape[0]
        assert Sigma.shape == (k, k)

        self.rng = rng
        self.k = k

        if jitter > 0.0:
            Sigma = Sigma + jitter * np.eye(k)

        self.Sigma = Sigma
        self.L = np.linalg.cholesky(Sigma)

    def step(self):
        self.x = self.L @ self.rng.standard_normal(self.k)
        return self.x

    def sample(self, N: int):
        Z = self.rng.standard_normal((N, self.k))
        return Z @ self.L.T


def gauss_markov_noise(N: int, dt: float, tau: Union[float, np.ndarray], Sigma: np.ndarray,
            x0: Optional[np.ndarray] = None, rng: Optional[np.random.Generator] = None,
            jitter: float = 0.0, from_stationary: bool = True,
) -> np.ndarray:
    """
    sigma is steady-state covariance
    Accurate discretization：x_{n+1} = Phi x_n + eta_n,  eta_n ~ N(0, Qd)
    in which: Phi = exp(-dt/tau) * I,  Qd = (1 - exp(-2 dt/tau)) * Sigma

    jitter: adds to the diag of Qd for steadibility
    from_stationary: if no x0 give, determines if the init value is drawn from a stationary distribution
    """
    if rng is None:
        rng = np.random.default_rng()
    k = Sigma.shape[0]
    assert Sigma.shape == (k, k)

    if type(tau) is not np.ndarray:
        tau = np.full(k, tau)

    # phi = np.exp(-dt / tau)
    # Phi = phi * np.eye(k)
    # Qd = (1.0 - phi**2) * Sigma
    phi = np.exp(-dt / tau)
    Phi = np.diag(phi)
    Qd = Sigma - (Phi @ Sigma @ Phi.T)
    if jitter > 0.0:
        Qd = Qd + jitter * np.eye(k)
    L = np.linalg.cholesky(Qd)

    x = np.empty((N, k))
    if x0 is None:
        if from_stationary:
            x[0] = rng.multivariate_normal(np.zeros(k), Sigma)
        else:
            x[0] = np.zeros(k)
    else:
        x[0] = np.asarray(x0)

    # equals: rng.multivariate_normal(np.zeros(k), Qd)
    # pre-cholesky saves computation
    for n in range(N - 1):
        x[n+1] = Phi @ x[n] + L @ rng.standard_normal(k)
    return x

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # one-var case
    N = 20000
    dt = 0.01
    tau = 20
    sigma = np.array([[0.03**2]])

    samples = gauss_markov_noise(N=N, dt=dt, tau=tau, Sigma=sigma)

    plt.plot(np.arange(samples.shape[0]), samples)
    plt.show()

    # tri-var case
    N = 9000
    dt = 0.1
    tau_vec = np.array([60.0, 60.0, 30.0])  # s for X,Y,N
    sigmas = np.array([5.0, 5.0, 1.0])     # N, N, N·m (steady-state std)

    Sigma = np.diag(sigmas**2)

    samples = gauss_markov_noise(N=N, dt=dt, tau=tau_vec, Sigma=Sigma, from_stationary=False)
    Fxs = samples[:,0]
    Fys = samples[:,1]
    Nrs = samples[:,2]

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(np.arange(samples.shape[0])*dt, Fxs)
    axs[1].plot(np.arange(samples.shape[0])*dt, Fys)
    axs[2].plot(np.arange(samples.shape[0])*dt, Nrs)

    plt.show()

    # one-var case in class
    N = 200
    dt = 0.01
    tau = 20
    sigma = np.array([[0.03**2]])

    samples = []
    g_samples = []
    noise_gen = GaussMarkovNoiseGenerator(dt=dt, tau=tau, Sigma=sigma)
    gauss_gen = GaussianNoiseGenerator(Sigma=sigma)
    for _ in range(N):
        samples.append(noise_gen.step())
        g_samples.append(gauss_gen.step())
    
    samples = np.hstack(samples)
    g_samples = np.hstack(g_samples)

    plt.plot(np.arange(samples.shape[0]), samples)
    plt.plot(np.arange(samples.shape[0]), g_samples)
    plt.show()