# encoding: utf-8
# author: vectorwang@hotmail.com
# license: MIT

r"""
 _           
| | ___  ___ 
| |/ _ \/ __|
| | (_) \__ \
|_|\___/|___/

"""
import numpy as np

from typing import Tuple, Iterable
from .pid import PID

class LOSGuider:
    """
    """
    def __init__(
            self,
            waypoints: Iterable[Tuple[float, float]],
            reached_threshold: float,
    ):
        """
        """
        self.reference_path = waypoints[:]
        self.reached_threshold = reached_threshold

    def calc_desired_direction(
            self,
            cur_pos: Tuple[float, float],
            tgt_pos: Tuple[float, float],
    ) -> float:
        cur_x, cur_y = cur_pos
        tgt_x, tgt_y = tgt_pos
        delta_x = tgt_x - cur_x
        delta_y = tgt_y - cur_y

        psi = np.arctan2(delta_y, delta_x)

        return psi

    def has_reached(
            self,
            cur_pos: Tuple[float, float],
            tgt_pos: Tuple[float, float],
    ) -> bool:
        cur_pos = np.array(cur_pos)
        tgt_pos = np.array(tgt_pos)

        distance_sq = np.sum(np.power(tgt_pos-cur_pos, 2))
        return distance_sq <= np.power(self.reached_threshold, 2)

    def step(
            self,
            cur_pos: Tuple[float, float],
            cur_psi: float,
    ) -> Tuple[bool, float]:
        """
        returns: is_ended, psi_err
        """
        cur_target = self.reference_path[0]
        while self.has_reached(cur_pos, cur_target):
            del(self.reference_path[0])
            if len(self.reference_path) == 0:
                # if all targets reached, return True
                return (True, 0)
            else:
                # else find and check next target
                cur_target = self.reference_path[0]
        
        desired_psi = self.calc_desired_direction(cur_pos, cur_target)
        psi_err = desired_psi - cur_psi
        psi_err %= 2*np.pi
        psi_err = psi_err - 2*np.pi if psi_err > np.pi else psi_err

        return False, psi_err

    @property
    def current_target(self):
        return self.reference_path[0]