import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.KT import FirstOrderResponse, sample_hydro_params, NoisyFirstOrderResponse
from identification.least_square_methods import LeastSquareFirstOrderNonLinearResponse, RecursiveLeastSquareFirstOrderNonLinearResponse
from demos.kt_zigzag import kt_zigzag



train_data = np.load("./demos/KT_noisy_zigzag_25_5_0.01.npy")
time_step = 0.01
# train_data before this index, will be treated as init data
# after this index, will be treated update data, which will be used for recursive update
update_index = 200

identifier = RecursiveLeastSquareFirstOrderNonLinearResponse(time_step=time_step)
result = identifier.identificate(train_data=train_data[:, :update_index])
K = result["K"]
T = result["T"]
alpha = result["alpha"]
print(K, T, alpha)

thetas = []

result = identifier.identificate(train_data=train_data[:, update_index:], thetas=thetas)
K = result["K"]
T = result["T"]
alpha = result["alpha"]

print(K, T, alpha)

# thetas is now a 3xn array
# a1, a2, b1
thetas = np.hstack(thetas)

# plot src
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.plot(train_data[1, :], train_data[0, :], label="Train data")

# plot prediction
# init simulation
u = sample_hydro_params["u"]
kt_simul = VesselSimulator(
    hydro_params={"K":K, "T":T, "alpha":alpha, "u":u},
    time_step=time_step,
    model=FirstOrderResponse,
    init_state=np.array([0, 0, 0, u, 0, 0]).reshape([6, 1]),
)

states = kt_zigzag(
    simulator=kt_simul,
    zigzag_degrees=25,
    sample_delta=5,
    sample_u=u,
)

ax.plot(states[1, :], states[0, :], label="Prediction")
ax.legend()

plt.show()

# plot convergence
fig, axs = plt.subplots(3, 1)

def plot_convergence(ax, data, tag):
    ax.plot(np.arange(len(data)), data, label=tag)
    ax.set_xlabel("Iteration")
    ax.legend()

plot_convergence(axs[0], thetas[0], "a1")
plot_convergence(axs[1], thetas[1], "a2")
plot_convergence(axs[2], thetas[2], "b1")
plt.show()