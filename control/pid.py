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
from collections import deque
from typing import Optional


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
        self.store_error(current_error)

        control_signal = self.calc_p() + self.calc_d() + self.calc_i()

        if self.min_saturation is not None and control_signal < self.min_saturation:
            return self.min_saturation
        elif self.max_saturation is not None and control_signal > self.max_saturation:
            return self.max_saturation
        else:
            return control_signal