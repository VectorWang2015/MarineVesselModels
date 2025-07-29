import numpy as np
from matplotlib import pyplot as plt


def original_euler(
        fn_p_y,
        y_0,
        t_0,
        h,
):
    """
    input:
        fn_p_y: function(y, t)
            takes in the current fn value y and time t,
            returns partial of the fn value y at time t
        y_0: init value / current value
        t_0: init time in [seconds]
        h: time step size in [seconds]
    returns:
        predicted y
    """
    # find a better slope
    k1 = fn_p_y(y_0, t_0)
    k = k1

    return y_0+k*h


def original_runge_kutta(
        fn_p_y,
        y_0,
        t_0,
        h,
):
    """
    input:
        fn_p_y: function(y, t)
            takes in the current fn value y and time t,
            returns partial of the fn value y at time t
        y_0: init value / current value
        t_0: init time in [seconds]
        h: time step size in [seconds]
    returns:
        predicted y
    """
    # find a better slope
    k1 = fn_p_y(y_0, t_0)
    k2 = fn_p_y(y_0+h/2*k1, t_0+h/2)
    k3 = fn_p_y(y_0+h/2*k2, t_0+h/2)
    k4 = fn_p_y(y_0+h*k3, t_0+h)
    k = (k1 + 2*k2 + 2*k3 + k4)/6

    return y_0+k*h


def time_invariant_simulator_euler(
        fn_p_state,
        state,
        tau,
        h,
):
    """
    assumes that input remains the same in the time step
    a typical time_invariant simulator takes in the current state and tau to predict
    input:
        fn_p_state: function(state, tau)
            takes in the current state and input tau,
            returns partial of the state
        state: current_state
        h: time step size in [seconds]
    returns:
        predicted state after time h
    """
    # find a better slope
    k1 = fn_p_state(state, tau)
    k = k1

    return state+k*h


def time_invariant_simulator_runge_kutta(
        fn_p_state,
        state,
        tau,
        h,
):
    """
    assumes that input remains the same in the time step
    a typical time_invariant simulator takes in the current state and tau to predict
    input:
        fn_p_state: function(state, tau)
            takes in the current state and input tau,
            returns partial of the state
        state: current_state
        h: time step size in [seconds]
    returns:
        predicted state after time h
    """
    # find a better slope
    k1 = fn_p_state(state, tau)
    k2 = fn_p_state(state+h/2*k1, tau)
    k3 = fn_p_state(state+h/2*k2, tau)
    k4 = fn_p_state(state+h*k3, tau)
    k = (k1 + 2*k2 + 2*k3 + k4)/6

    return state+k*h


if __name__ == "__main__":
    # plot sin
    sin_ts = np.arange(0, 7, 0.01)
    sin_ys = np.sin(sin_ts)
    plt.plot(sin_ts, sin_ys)

    # test partial fn: p_sin is cos
    def p_sin_y(y, t):
        return np.cos(t)

    # step parameters
    h = 0.4
    count = 18
    y_0 = 0

    # test code for euler
    euler_ys = [y_0]
    for i in range(count):
        t = h*i
        y = euler_ys[-1]
        y = original_euler(p_sin_y, y, t, h)
        euler_ys.append(y)

    plt.scatter(np.arange(0, count*h+h, h), euler_ys, marker="x", color="navy", label="Euler")

    # test code for runge-kutta
    rk_ys = [y_0]
    for i in range(count):
        t = h*i
        y = rk_ys[-1]
        y = original_runge_kutta(p_sin_y, y, t, h)
        rk_ys.append(y)

    plt.scatter(np.arange(0, count*h+h, h), rk_ys, marker="x", color="red", label="Runge Kutta")

    plt.legend()
    plt.show()