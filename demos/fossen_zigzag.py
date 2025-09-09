import numpy as np
from matplotlib import pyplot as plt

from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster
from MarineVesselModels.Fossen import sample_b_2, sample_hydro_params_2, sample_thrust_2, Fossen


"""
# surge test, with time_step 0.1
us = []
time_step = 0.1
total_time = 20
total_steps = int(total_time / time_step)

base_N = 50.0
base_delta_N = 10.0

current_state = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])
u = current_state[3][0]
us.append(u)

simulator = VesselSimulator(
    sample_hydro_params_2,
    time_step=time_step,
    model=Fossen,
    init_state=current_state,
)
thruster = NaiveDoubleThruster(b=sample_b_2)

for i in range(total_steps):
    # iterate, find converged velo under base_N
    tau = thruster.newton_to_tau(base_N, base_N)
    current_state = simulator.step(tau)
    
    u = current_state[3][0]
    us.append(u)

print("Converged u under ({}, {})N and timestep {}: {}.".format(base_N, base_N, time_step, us[-1]))

# surge test, with time_step 0.01, time_step convergence check
us = []
time_step = 0.01
total_time = 20
total_steps = int(total_time / time_step)

base_N = 50.0
base_delta_N = 10.0

current_state = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])
u = current_state[3][0]
us.append(u)

simulator = VesselSimulator(
    sample_hydro_params_2,
    time_step=time_step,
    model=Fossen,
    init_state=current_state,
)
thruster = NaiveDoubleThruster(b=sample_b_2)

for i in range(total_steps):
    # iterate, find converged velo under base_N
    tau = thruster.newton_to_tau(base_N, base_N)
    current_state = simulator.step(tau)
    
    u = current_state[3][0]
    us.append(u)

print("Converged u under ({}, {})N and timestep {}: {}.".format(base_N, base_N, time_step, us[-1]))

plt.plot(np.arange(len(us)), us)
plt.show()
"""

# zigzag 10
# converged u under base thrust 50N: 0.659761
time_step = 0.01
total_time = 20
total_steps = int(total_time / time_step)

init_psi = 0
base_N = 50.0
base_delta_N = 20.0
zigzag_degrees = 10.0 / 180 * np.pi

current_state = np.array([0, 0, init_psi, 0.659761, 0, 0]).reshape([6, 1])
states = current_state.copy()

simulator = VesselSimulator(
    sample_hydro_params_2,
    time_step=time_step,
    model=Fossen,
    init_state=current_state,
)
thruster = NaiveDoubleThruster(b=sample_b_2)

# turn right
tgt_psi = init_psi + zigzag_degrees
left_thr = base_N + base_delta_N
right_thr = base_N - base_delta_N
tau = thruster.newton_to_tau(left_thr, right_thr)
while current_state[2][0] < tgt_psi:
    current_state = simulator.step(tau)
    states = np.hstack((states, current_state))
# turn left
tgt_psi = init_psi - zigzag_degrees
left_thr = base_N - base_delta_N
right_thr = base_N + base_delta_N
tau = thruster.newton_to_tau(left_thr, right_thr)
while current_state[2][0] > tgt_psi:
    current_state = simulator.step(tau)
    states = np.hstack((states, current_state))
# turn right
tgt_psi = init_psi + zigzag_degrees
left_thr = base_N + base_delta_N
right_thr = base_N - base_delta_N
tau = thruster.newton_to_tau(left_thr, right_thr)
while current_state[2][0] < tgt_psi:
    current_state = simulator.step(tau)
    states = np.hstack((states, current_state))

# fig, ax = plt.subplots()
# ax.set_aspect("equal")
# ax.plot(states[1], states[0])
# plt.show()
np.save("Fossen_zigzag_50_20_10.npy", states)


# zigzag 20
# converged u under base thrust 50N: 0.659761
time_step = 0.01
total_time = 20
total_steps = int(total_time / time_step)

init_psi = 0
base_N = 50.0
base_delta_N = 20.0
zigzag_degrees = 20.0 / 180 * np.pi

current_state = np.array([0, 0, init_psi, 0.659761, 0, 0]).reshape([6, 1])
states = current_state.copy()

simulator = VesselSimulator(
    sample_hydro_params_2,
    time_step=time_step,
    model=Fossen,
    init_state=current_state,
)
thruster = NaiveDoubleThruster(b=sample_b_2)

# turn right
tgt_psi = init_psi + zigzag_degrees
left_thr = base_N + base_delta_N
right_thr = base_N - base_delta_N
tau = thruster.newton_to_tau(left_thr, right_thr)
while current_state[2][0] < tgt_psi:
    current_state = simulator.step(tau)
    states = np.hstack((states, current_state))
# turn left
tgt_psi = init_psi - zigzag_degrees
left_thr = base_N - base_delta_N
right_thr = base_N + base_delta_N
tau = thruster.newton_to_tau(left_thr, right_thr)
while current_state[2][0] > tgt_psi:
    current_state = simulator.step(tau)
    states = np.hstack((states, current_state))
# turn right
tgt_psi = init_psi + zigzag_degrees
left_thr = base_N + base_delta_N
right_thr = base_N - base_delta_N
tau = thruster.newton_to_tau(left_thr, right_thr)
while current_state[2][0] < tgt_psi:
    current_state = simulator.step(tau)
    states = np.hstack((states, current_state))

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.plot(states[1], states[0])
plt.show()

fig, axs = plt.subplots(4, 1)
axs[0].plot(np.arange(states.shape[1])*time_step, states[2]/np.pi*180, label="$\psi$")
axs[0].legend()
axs[1].plot(np.arange(states.shape[1])*time_step, states[5], label="$\omega$")
axs[1].legend()
axs[2].plot(np.arange(states.shape[1])*time_step, states[3], label="$u$")
axs[2].legend()
axs[3].plot(np.arange(states.shape[1])*time_step, states[4], label="$v$")
axs[3].legend()
plt.show()

np.save("Fossen_zigzag_50_20_20.npy", states)