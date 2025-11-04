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
            output_err_flag: bool = True,
    ):
        """
        """
        self.reference_path = waypoints[:]
        self.reached_threshold = reached_threshold
        self.output_err_flag = output_err_flag

        self.former_waypoint = None
        self.current_waypoint = None
        self.cur_pos = None

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

    def update_waypoint(self):
        if self.former_waypoint is None:
            self.former_waypoint = self.cur_pos
        else:
            self.former_waypoint = self.current_waypoint
        self.current_waypoint = self.reference_path[0]

    @property
    def current_target(self):
        """
        in naive LOS, the next waypoint is the next target
        """
        return self.current_waypoint

    def step(
            self,
            cur_pos: Tuple[float, float],
            cur_psi: float,
    ) -> Tuple[bool, float]:
        """
        returns: is_ended, psi_err
        """
        self.cur_pos = cur_pos

        # init waypoint if first time step is called
        if self.current_waypoint is None:
            self.update_waypoint()

        # find 
        while self.has_reached(self.cur_pos, self.current_waypoint):
            del(self.reference_path[0])
            if len(self.reference_path) == 0:
                # if all waypoints reached, return True
                if self.output_err_flag:
                    return (True, 0)
                else:
                    return (True, None)
            else:
                # else set/check next waypoint
                self.update_waypoint()
        
        desired_psi = self.calc_desired_direction(cur_pos, self.current_target)

        if self.output_err_flag:
            psi_err = desired_psi - cur_psi
            psi_err %= 2*np.pi
            psi_err = psi_err - 2*np.pi if psi_err > np.pi else psi_err
            return False, psi_err
        else:
            return False, desired_psi


class FixedDistLOSGuider(LOSGuider):
    """
    """
    def __init__(
            self,
            waypoints: Iterable[Tuple[float, float]],
            reached_threshold: float,
            los_dist: float,
            output_err_flag: bool = True,
    ):
        """
        """
        super().__init__(waypoints, reached_threshold, output_err_flag)

        self.los_dist = los_dist

    @property
    def current_target(self):
        """
        in fixed LOS, current target is:
            * such is the perpendicular point from cur_pos to the current way line (if dist >= los_dist)
            * current (next) way point (if way point within los_dist)
            * a point that is on the way line, and is los_dist from cur_pos (otherwise)
        """
        raise Exception
        return self.current_waypoint

    def step(
            self,
            cur_pos: Tuple[float, float],
            cur_psi: float,
    ) -> Tuple[bool, float]:
        """
        returns: is_ended, psi_err
        """
        self.cur_pos = cur_pos

        # init waypoint if first time step is called
        if self.current_waypoint is None:
            self.update_waypoint()

        # find 
        while self.has_reached(self.cur_pos, self.current_waypoint):
            del(self.reference_path[0])
            if len(self.reference_path) == 0:
                # if all waypoints reached, return True
                if self.output_err_flag:
                    return (True, 0)
                else:
                    return (True, None)
            else:
                # else set/check next waypoint
                self.update_waypoint()
        
        desired_psi = self.calc_desired_direction(cur_pos, self.current_target)

        if self.output_err_flag:
            psi_err = desired_psi - cur_psi
            psi_err %= 2*np.pi
            psi_err = psi_err - 2*np.pi if psi_err > np.pi else psi_err
            return False, psi_err
        else:
            return False, desired_psi
