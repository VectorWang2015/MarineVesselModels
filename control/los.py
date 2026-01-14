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

from typing import Tuple, Iterable, Optional
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
        self.cur_psi = None

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

    def update_cur_status(
            self,
            cur_pos: Tuple[float, float],
            cur_psi: float,
    ) -> bool:
        """
        :return is_ended: True if all waypoints reached, else False
        """
        self.cur_pos = cur_pos
        self.cur_psi = cur_psi

        # init waypoint if first time step is called
        if self.current_waypoint is None:
            self.update_waypoint()

        # find
        while self.has_reached(self.cur_pos, self.current_waypoint):
            del (self.reference_path[0])
            if len(self.reference_path) == 0:
                # if all waypoints reached, return True
                return True
            else:
                # else set/check next waypoint
                self.update_waypoint()
        return False

    def calc_psi_err(
            self,
            desired_psi: float,
    ) -> float:
        """
        :return psi_err: float within [-pi, pi]
        """
        psi_err = desired_psi - self.cur_psi
        psi_err = (psi_err + np.pi) % (2*np.pi) - np.pi
        return psi_err

    def step(
            self,
            cur_pos: Tuple[float, float],
            cur_psi: float,
    ) -> Tuple[bool, float]:
        """
        returns: is_ended, psi_err
        """
        has_ended = self.update_cur_status(cur_pos, cur_psi)
        if has_ended:
            if self.output_err_flag:
                return (True, 0)
            else:
                return (True, None)

        desired_psi = self.calc_desired_direction(cur_pos, self.current_target)

        if self.output_err_flag:
            psi_err = self.calc_psi_err(desired_psi)
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

    def calc_los_target(
            self,
    ):
        """
        return a target point
        segment should be a line from the desired trajectory, a tuple of start and end point
        should be guided to start or end point if each pt on the desired path is beyond the guide distance
        """
        guide_line_dist = self.los_dist
        line_pt1 = np.array(self.former_waypoint)
        line_pt2 = np.array(self.current_waypoint)
        current_pos = self.cur_pos

        l2 = np.sum((line_pt2-line_pt1)**2)

        line_vec = (line_pt2 - line_pt1) / np.sqrt(l2)

        proportion = np.dot(line_pt2-line_pt1, current_pos-line_pt1) / l2
        perpendicular_pt = proportion * (line_pt2 - line_pt1) + line_pt1
        dist = np.linalg.norm(perpendicular_pt-current_pos)

        #breakpoint()
        if dist >= guide_line_dist:
            if proportion <= 0:
                return line_pt1
            elif proportion >= 1:
                return line_pt2
            else:
                return perpendicular_pt

        elif dist < guide_line_dist:
            remaining_length = np.sqrt(guide_line_dist**2 - dist**2)
            remaining_vec = remaining_length * line_vec
            target_pt = perpendicular_pt + remaining_vec
            target_pt_proportion = np.dot(target_pt - line_pt1, line_vec) / np.sqrt(l2)
            if target_pt_proportion <= 0:
                return line_pt1
            elif target_pt_proportion >= 1:
                return line_pt2
            else:
                return target_pt

        else:
            raise Exception

    @property
    def current_target(self):
        """
        in fixed LOS, current target is:
            * such is the perpendicular point from cur_pos to the current way line (if dist >= los_dist)
            * current (next) way point (if way point within los_dist)
            * a point that is on the way line, and is los_dist from cur_pos (otherwise)
        """
        return self.calc_los_target()


class DynamicDistLOSGuider(FixedDistLOSGuider):
    """
    """

    def __init__(
            self,
            waypoints: Iterable[Tuple[float, float]],
            reached_threshold: float,
            forward_dist: float,
            output_err_flag: bool = True,
    ):
        """
        """
        self.reference_path = waypoints[:]
        self.reached_threshold = reached_threshold
        self.output_err_flag = output_err_flag
        self.forward_dist = forward_dist

        self.former_waypoint = None
        self.current_waypoint = None
        self.cur_pos = None

    def calc_los_target(
            self,
    ):
        """
        return a target point
        guider logic see: <Path following control system for a tanker ship model>
        """
        remaining_length = self.forward_dist
        line_pt1 = np.array(self.former_waypoint)
        line_pt2 = np.array(self.current_waypoint)
        current_pos = self.cur_pos

        l2 = np.sum((line_pt2-line_pt1)**2)

        line_vec = (line_pt2 - line_pt1) / np.sqrt(l2)

        proportion = np.dot(line_pt2-line_pt1, current_pos-line_pt1) / l2
        perpendicular_pt = proportion * (line_pt2 - line_pt1) + line_pt1
        #dist = np.linalg.norm(perpendicular_pt-current_pos)

        #breakpoint()
        #remaining_length = np.sqrt(guide_line_dist**2 - dist**2)
        remaining_vec = remaining_length * line_vec
        target_pt = perpendicular_pt + remaining_vec
        target_pt_proportion = np.dot(target_pt - line_pt1, line_vec) / np.sqrt(l2)
        if target_pt_proportion <= 0:
            return line_pt1
        elif target_pt_proportion >= 1:
            return line_pt2
        else:
            return target_pt


class AdaptiveLOSGuider(DynamicDistLOSGuider):
    """
    Adaptive LOS (ALOS) guider with sideslip compensation.

    Based on Fossen 2023: ψ_d = π_h - β_hat - atan(y_e/Δ)

    Algorithm:
    1. Compute path-tangential angle π_h = atan2(y_{i+1} - y_i, x_{i+1} - x_i)
    2. Compute signed cross-track error y_e in path-tangential frame
    3. Desired heading: ψ_d = π_h - β_hat - atan(y_e/Δ)
    4. Sideslip adaptation: β_hat_dot = γ·(Δ·y_e)/√(Δ² + y_e²)
    """

    def __init__(
            self, waypoints, reached_threshold, forward_dist,
            dt,
            gamma=0.0006, beta_hat0=0.0,
            output_err_flag=True,
            reset_beta_on_segment_change=False
    ):
        super().__init__(waypoints, reached_threshold, forward_dist, output_err_flag)
        assert forward_dist > 0, "forward_dist must be positive"
        self.gamma = gamma
        self.beta_hat0 = beta_hat0
        self.beta_hat = beta_hat0
        self.dt = dt
        self.reset_beta_on_segment_change = reset_beta_on_segment_change

    def update_cur_status(
            self,
            cur_pos: Tuple[float, float],
            cur_psi: float,
    ) -> bool:
        """
        :return is_ended: True if all waypoints reached, else False
        """
        return super().update_cur_status(cur_pos, cur_psi)

    def calc_pi_h(self):
        """
        Compute path-tangential angle π_h for current segment.

        !!! Should be called after waypoints updated

        π_h = atan2(y_{i+1} - y_i, x_{i+1} - x_i)

        Returns:
        --------
        float : π_h in radians, normalized to [0, 2π)
        """
        line_pt1 = self.former_waypoint
        line_pt2 = self.current_waypoint
        delta_x = line_pt2[0] - line_pt1[0]
        delta_y = line_pt2[1] - line_pt1[1]
        return np.arctan2(delta_y, delta_x)

    def calc_cross_track_error(self):
        """
        Compute signed cross-track error y_e in path-tangential frame.

        y_e = (p × s) / ‖s‖ where p = position vector, s = segment vector
        Positive y_e indicates port side deviation.

        Returns:
        --------
        float : y_e in meters
        """
        line_pt1 = self.former_waypoint
        #line_pt2 = self.current_waypoint
        pos = self.cur_pos
        pi_h = self.calc_pi_h()

        dx = pos[0] - line_pt1[0]
        dy = pos[1] - line_pt1[1]
        cross_track_err = -np.sin(pi_h)*dx + np.cos(pi_h)*dy

        return cross_track_err

    def calc_desired_direction(self, cur_pos, tgt_pos=None):
        """
        Override to compute ALOS desired heading.

        !!! Needs to be called after pos/waypoints updated.

        ψ_d = π_h - β_hat - atan(y_e/Δ)

        Parameters:
        -----------
        cur_pos : Tuple[float, float]
            Current position (x, y)
        tgt_pos : Tuple[float, float], optional
            Target position (ignored in ALOS)

        Returns:
        --------
        float : Desired heading ψ_d in radians
        """
        D = self.forward_dist

        pi_h = self.calc_pi_h()
        y_e = self.calc_cross_track_error()
        dot_beta = self.gamma * D * y_e / \
                np.sqrt(D**2 + y_e**2)
        self.beta_hat += dot_beta * self.dt

        return pi_h - self.beta_hat - np.arctan(y_e / D)

    def update_waypoint(self):
        """
        Override to reset beta_hat on segment transitions if enabled.
        """
        prev_waypoint = self.current_waypoint
        super().update_waypoint()
        if self.reset_beta_on_segment_change and prev_waypoint is not None:
            self.beta_hat = self.beta_hat0

    @property
    def current_target(self):
        raise Exception("ALOS calculation is not based on a hypo target.")

    def step(self, cur_pos, cur_psi):
        """
        Override step method for ALOS adaptation.

        Parameters:
        -----------
        cur_pos : Tuple[float, float]
            Current position (x, y)
        cur_psi : float
            Current heading (rad)

        Returns:
        --------
        Tuple[bool, float] : (is_ended, psi_err/desired_psi)
        """
        has_ended = self.update_cur_status(cur_pos, cur_psi)
        if has_ended:
            if self.output_err_flag:
                return (True, 0)
            else:
                return (True, None)

        desired_psi = self.calc_desired_direction(cur_pos)

        if self.output_err_flag:
            psi_err = self.calc_psi_err(desired_psi)
            return False, psi_err
        else:
            return False, desired_psi
