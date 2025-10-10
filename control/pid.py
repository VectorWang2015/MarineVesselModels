# encoding: utf-8
# author: vectorwang@hotmail.com
# license: MIT

r"""
       _     _ 
 _ __ (_) __| |
| '_ \| |/ _` |
| |_) | | (_| |
| .__/|_|\__,_|
|_|
"""
from math import pi
from collections import deque
from typing import Optional

def wrap_pi(psi):
    return (psi + pi) % (2*pi) - pi


class PID:
    def __init__(
                self,
                kp: float,
                ki: float,
                kd: float,
                buffer_size: int=200,
                min_saturation: Optional[float] = None,
                max_saturation: Optional[float] = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.buffer_size = buffer_size
        self.error_buffer = deque(maxlen=self.buffer_size)
        self.desire_value = None

        self.min_saturation = min_saturation
        self.max_saturation = max_saturation
    
    def store_error(self, error: float):
        self.error_buffer.append(error)
        if len(self.error_buffer) > self.buffer_size:
            del(self.error_buffer[0])

    def calc_p(self) -> float:
        return self.kp * self.error_buffer[-1]

    def calc_i(self) -> float:
        return self.ki * sum(self.error_buffer) / self.buffer_size

    def calc_d(self) -> float:
        if len(self.error_buffer) < 2:
            return 0
        return self.kd * (self.error_buffer[-1] - self.error_buffer[-2])

    def change_desire_value(self, desire_value):
        self.desire_value = desire_value
        # clear buffer if desire changed
        #self.error_buffer = deque(maxlen=self.buffer_size)

    def control(
            self,
            desire_value: Optional[float] = None,
            current_value: Optional[float] = None,
            error: Optional[float] = None,
    ) -> float:
        if error is not None:
            current_error = error
        else:
            current_error = desire_value - current_value
            if desire_value != self.desire_value:
                self.change_desire_value()
        self.store_error(current_error)

        control_signal = self.calc_p() + self.calc_d() + self.calc_i()

        if self.min_saturation is not None and control_signal < self.min_saturation:
            return self.min_saturation
        elif self.max_saturation is not None and control_signal > self.max_saturation:
            return self.max_saturation
        else:
            return control_signal


class PIDAW:  # PID with Anti-Windup & derivative filter
    def __init__(self, kp, ki, kd, dt,
                u_min=None, u_max=None,
                tau_d=0.05,     # low-pas filter const
                k_aw=0.5,       # anti-windup: back calculation rate 0~1
                beta=1.0      # setpoint weighting for P
    ):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.u_min, self.u_max = u_min, u_max
        self.tau_d = tau_d
        self.k_aw = k_aw
        self.beta = beta

        self.I = 0.0
        self.y_f = None     # filtered sample
        self.d = 0.0
        self.u_prev = 0.0
        self.sp_prev = None

    def reset(self, u_hold=None):
        self.I = 0.0
        self.y_f = None
        self.d = 0.0
        if u_hold is not None:
            self.u_prev = u_hold

    def bumpless_preset_I(self, e, d_term, u_target):
        # not enabled yet
        if self.ki > 0.0:
            self.I = (u_target - self.kp*e - d_term) / self.ki

    def step(self, sp, y, y_is_angle=False):
        e = wrap_pi(sp - y) if y_is_angle else (sp - y)

        # low-pass filt
        if self.y_f is None:
            self.y_f = y
        alpha = self.dt / (self.tau_d + self.dt)
        self.y_f += alpha * (y - self.y_f)

        # diff term calculation
        dy = (self.y_f - getattr(self, "_y_f_prev", self.y_f)) / self.dt
        self._y_f_prev = self.y_f
        d_term = - self.kd * dy

        # P + I + D
        u_raw = self.kp * (self.beta * e) + self.I + d_term

        # saturation
        u_sat = u_raw
        if self.u_min is not None: u_sat = max(self.u_min, u_sat)
        if self.u_max is not None: u_sat = min(self.u_max, u_sat)

        # anti-windup: back calculation
        aw = u_sat - u_raw
        self.I += (self.ki * e + self.k_aw * aw) * self.dt

        self.u_prev = u_sat
        return u_sat


class DoubleLoopHeadingPID:
    def __init__(self, dt,
                # outer loop (psi -> r)
                psi_kp, psi_ki, psi_kd,
                r_ref_lim,           # staturation for r (rad/s)
                # inner loop (r -> controller)
                r_kp, r_ki, r_kd,
                u_lim,               # satuaration for control value
                r_ref_slew=None,     # slew rate for r (rad/s^2)
                u_slew=None,         # slew rate for u
    ):
        self.dt = dt
        self.r_ref_lim = r_ref_lim
        self.u_lim = u_lim
        self.r_ref_slew = r_ref_slew
        self.u_slew = u_slew

        self.psi_pid = PIDAW(psi_kp, psi_ki, psi_kd, dt,
                             u_min=-r_ref_lim, u_max=r_ref_lim,
                             tau_d=0.1, k_aw=0.5, beta=1.0)

        self.r_pid   = PIDAW(r_kp, r_ki, r_kd, dt,
                             u_min=-u_lim, u_max=u_lim,
                             tau_d=0.05, k_aw=0.5, beta=1.0)

        self._r_ref_prev = 0.0
        self._u_prev = 0.0

    def _slew(self, x_prev, x_cmd, slew_limit):
        if slew_limit is None:
            return x_cmd
        max_step = slew_limit * self.dt
        return max(min(x_cmd, x_prev + max_step), x_prev - max_step)

    def step(self, psi_ref, psi, r):
        r_ref_cmd = self.psi_pid.step(psi_ref, psi, y_is_angle=True)
        r_ref = self._slew(self._r_ref_prev, r_ref_cmd, self.r_ref_slew)
        r_ref = max(min(r_ref, self.r_ref_lim), -self.r_ref_lim)
        self._r_ref_prev = r_ref

        u_cmd = self.r_pid.step(r_ref, r, y_is_angle=False)
        u = self._slew(self._u_prev, u_cmd, self.u_slew)
        u = max(min(u, self.u_lim), -self.u_lim)
        self._u_prev = u

        return u, r_ref