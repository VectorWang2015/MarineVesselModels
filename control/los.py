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
    Naive Line-of-Sight (LOS) guider.

    Guides vessel directly toward the next waypoint without lookahead distance.
    """

    def __init__(
            self,
            waypoints: Iterable[Tuple[float, float]],
            reached_threshold: float,
            output_err_flag: bool = True,
    ):
        """
        :param waypoints: List of (x, y) coordinates defining the desired path
        :param reached_threshold: Distance threshold for considering a waypoint reached (meters)
        :param output_err_flag: If True, step() returns heading error; if False, returns desired heading
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
        """
        Compute desired heading angle to target position.

        :param cur_pos: Current (x, y) position
        :param tgt_pos: Target (x, y) position
        :return: Desired heading angle in radians
        """
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
        """
        Check if current position is within reached_threshold of target.

        :param cur_pos: Current (x, y) position
        :param tgt_pos: Target (x, y) position
        :return: True if squared distance <= reached_threshold ** 2
        """
        cur_pos = np.array(cur_pos)
        tgt_pos = np.array(tgt_pos)

        distance_sq = np.sum(np.power(tgt_pos-cur_pos, 2))
        return distance_sq <= np.power(self.reached_threshold, 2)

    def update_waypoint(self):
        """
        Advance to next waypoint in reference path.

        Updates former_waypoint and current_waypoint.
        Called when current waypoint is reached.
        """
        if self.former_waypoint is None:
            self.former_waypoint = self.cur_pos
        else:
            self.former_waypoint = self.current_waypoint
        self.current_waypoint = self.reference_path[0]

    @property
    def current_target(self):
        """
        Get current target position.

        :return: Current target (x, y) tuple
        """
        return self.current_waypoint

    def update_cur_status(
            self,
            cur_pos: Tuple[float, float],
            cur_psi: float,
    ) -> bool:
        """
        Update guider status with current position and heading.

        :param cur_pos: Current (x, y) position
        :param cur_psi: Current heading angle (radians)
        :return: True if all waypoints reached, else False
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
        Compute heading error normalized to [-π, π].

        :param desired_psi: Desired heading angle (radians)
        :return: Heading error in radians, normalized to [-π, π]
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
        Compute guidance output for current state.

        :param cur_pos: Current (x, y) position
        :param cur_psi: Current heading angle (radians)
        :return: Tuple (is_ended, psi_err/desired_psi) where:
                 is_ended = True if all waypoints reached,
                 psi_err/desired_psi = heading error if output_err_flag True else desired heading
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
    Fixed lookahead distance LOS guider.

    Guides vessel toward a point on the path segment that is a fixed lookahead
    distance ahead of the vessel's perpendicular projection onto the segment.
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
        Calculate the LOS target point based on fixed lookahead distance.

        The target point is computed as:
        1. Project current position onto the current path segment
        2. If distance from projection >= los_dist, target is the projection point
        3. If distance < los_dist, target is a point los_dist ahead along the segment
        4. If target would be beyond segment endpoints, clamp to nearest endpoint

        :return: Target point as numpy array (x, y)
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
        Get current target position for fixed LOS guider.

        The target depends on vessel's distance from the path:
        - If perpendicular distance >= los_dist: target is perpendicular projection point
        - If waypoint within los_dist: target is the next waypoint
        - Otherwise: target is a point los_dist ahead along the segment

        :return: Current target position (x, y)
        """
        return self.calc_los_target()


class DynamicDistLOSGuider(FixedDistLOSGuider):
    """
    Dynamic lookahead distance LOS guider.

    Guides vessel toward a point on the path segment that is a fixed forward
    distance ahead of the vessel's perpendicular projection onto the segment,
    regardless of cross-track error.
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
        Calculate the LOS target point based on fixed forward distance.

        The target point is computed as a point forward_dist ahead along the path
        segment from the vessel's perpendicular projection onto the segment.
        If target would be beyond segment endpoints, clamp to nearest endpoint.

        :return: Target point as numpy array (x, y)
        """
        remaining_length = self.forward_dist
        line_pt1 = np.array(self.former_waypoint)
        line_pt2 = np.array(self.current_waypoint)
        current_pos = self.cur_pos

        l2 = np.sum((line_pt2-line_pt1)**2)

        line_vec = (line_pt2 - line_pt1) / np.sqrt(l2)

        proportion = np.dot(line_pt2-line_pt1, current_pos-line_pt1) / l2
        perpendicular_pt = proportion * (line_pt2 - line_pt1) + line_pt1
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
    """

    def __init__(
            self, waypoints, reached_threshold, forward_dist,
            dt,
            gamma=0.0006, beta_hat0=0.0,
            output_err_flag=True,
            reset_beta_on_segment_change=False
    ):
        """
        Initialize Adaptive LOS guider with sideslip compensation.

        :param waypoints: List of (x, y) coordinates defining the desired path
        :param reached_threshold: Distance threshold for considering a waypoint reached (meters)
        :param forward_dist: Lookahead distance Δ (meters)
        :param dt: Time step for sideslip adaptation (seconds)
        :param gamma: Adaptation gain γ
        :param beta_hat0: Initial sideslip estimate β̂₀ (radians)
        :param output_err_flag: If True, step() returns heading error; if False, returns desired heading
        :param reset_beta_on_segment_change: If True, reset β̂ to β̂₀ on segment transitions
        """
        super().__init__(
                waypoints, reached_threshold, forward_dist, output_err_flag)
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
        Update guider status with current position and heading.

        :param cur_pos: Current (x, y) position
        :param cur_psi: Current heading angle (radians)
        :return: True if all waypoints reached, else False
        """
        return super().update_cur_status(cur_pos, cur_psi)

    def calc_pi_h(self):
        """
        Compute path-tangential angle π_h for current segment.

        !!! Should be called after waypoints updated

        π_h = atan2(y_{i+1} - y_i, x_{i+1} - x_i)

        :return: Path-tangential angle π_h in radians, normalized to [0, 2π)
        """
        line_pt1 = self.former_waypoint
        line_pt2 = self.current_waypoint
        delta_x = line_pt2[0] - line_pt1[0]
        delta_y = line_pt2[1] - line_pt1[1]
        return np.arctan2(delta_y, delta_x)

    def calc_cross_track_error(self):
        """
        Compute signed cross-track error y_e in path-tangential frame.

        Positive y_e indicates starboard side deviation.

        :return: Cross-track error y_e in meters (positive = starboard side)
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

        :param cur_pos: Current position (x, y)
        :param tgt_pos: Target position (ignored in ALOS),
                left only for consistence
        :return: Desired heading ψ_d in radians
        """
        D = self.forward_dist

        pi_h = self.calc_pi_h()
        y_e = self.calc_cross_track_error()

        # use old estimate to generate control target,
        # to avoid estimation-control co-coupling
        desired_direction = pi_h - self.beta_hat - np.arctan(y_e / D)
        # update new estimation
        dot_beta = self.gamma * D * y_e / \
                np.sqrt(D**2 + y_e**2)
        self.beta_hat += dot_beta * self.dt

        return desired_direction

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

        :param cur_pos: Current position (x, y)
        :param cur_psi: Current heading (rad)
        :return: Tuple (is_ended, psi_err/desired_psi)
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


class EnhancedAdaptiveLOSGuider(AdaptiveLOSGuider):
    """
    Enhanced Adaptive LOS (EALOS) guider with improved sideslip adaptation.

    Extends AdaptiveLOSGuider with:
    1. β̂ clamping to [-β_max, +β_max] for physical realism
    2. Conditional integration: adapt only when |ψ_err| ≤ threshold
    3. Leakage term: σ·(β̂ - β̂₀) pulls estimate towards initial value
    4. Partial reset on segment changes: β̂ = α·β̂ + (1-α)·β̂₀
    """

    def __init__(
            self, waypoints, reached_threshold, forward_dist,
            dt,
            gamma=0.0006, sigma=0.0, beta_hat0=0.0,
            beta_max=np.deg2rad(30.0),
            psi_err_threshold=np.deg2rad(15.0),
            alpha=0.0,
            output_err_flag=True,
    ):
        """
        Initialize Enhanced Adaptive LOS guider.

        :param waypoints: List of (x, y) coordinates defining the desired path
        :param reached_threshold: Distance threshold for considering a waypoint reached (meters)
        :param forward_dist: Lookahead distance Δ (meters)
        :param dt: Time step for sideslip adaptation (seconds)
        :param gamma: Adaptation gain γ
        :param sigma: Leakage gain σ for sideslip estimate (default 0.0)
        :param beta_hat0: Initial sideslip estimate β̂₀ (radians)
        :param beta_max: Maximum allowed sideslip estimate magnitude (radians)
        :param psi_err_threshold: Heading error threshold for conditional integration (radians)
        :param alpha: Partial reset factor (0.0 = hard reset, 1.0 = no reset)
                Default hard reset (tuned through ablation in constant disturbance)
        :param output_err_flag: If True, step() returns heading error; if False, returns desired heading
        """
        super().__init__(
                waypoints, reached_threshold, forward_dist,
                dt, gamma, beta_hat0,
                output_err_flag=output_err_flag,
                reset_beta_on_segment_change=False  # Override parent's reset mechanism
        )
        assert sigma >= 0.0, "sigma must be non-negative"
        self.sigma = sigma
        self.beta_max = beta_max
        self.psi_err_threshold = psi_err_threshold
        self.alpha = alpha
        # Clamp initial beta_hat to [-beta_max, beta_max]
        self.beta_hat = np.clip(self.beta_hat, -self.beta_max, self.beta_max)

    def calc_desired_direction(self, cur_pos, tgt_pos=None):
        """
        Override to compute enhanced ALOS desired heading with improved adaptation.

        ψ_d = π_h - β_hat - atan(y_e/Δ)

        Adaptation law:
        dot_β̂ = γ·Δ·y_e/√(Δ² + y_e²) - σ·(β̂ - β̂₀)

        Adaptation features:
        1. Conditional integration based on heading error
        2. β̂ clamping to [-β_max, +β_max]
        3. Leakage term with gain σ pulls estimate towards β̂₀
        4. Partial reset on segment transitions

        :param cur_pos: Current position (x, y)
        :param tgt_pos: Target position (ignored in ALOS),
                left only for consistency
        :return: Desired heading ψ_d in radians
        """
        D = self.forward_dist
        pi_h = self.calc_pi_h()
        y_e = self.calc_cross_track_error()

        # 1) Use OLD estimate to generate control target (avoid estimation-control co-coupling)
        desired_direction = pi_h - self.beta_hat - np.arctan(y_e / D)

        # 2) Decide whether learning term is enabled (conditional integration)
        psi_err = self.calc_psi_err(desired_direction)
        learn_on = abs(psi_err) <= self.psi_err_threshold

        # 3) Compute learning term (gated) and leakage term (always on)
        learning_term = 0.0
        if learn_on:
            learning_term = self.gamma * D * y_e / np.sqrt(D**2 + y_e**2)

        leakage_term = self.sigma * (self.beta_hat - self.beta_hat0)

        # 4) Update beta_hat for NEXT step
        dot_beta = learning_term - leakage_term
        self.beta_hat += dot_beta * self.dt

        # 5) Clamp beta_hat after update
        self.beta_hat = np.clip(self.beta_hat, -self.beta_max, self.beta_max)

        return desired_direction

    def update_waypoint(self):
        """
        Override to apply partial reset on segment transitions.

        Partial reset: β̂ = α·β̂ + (1-α)·β̂₀
        """
        prev_waypoint = self.current_waypoint
        super().update_waypoint()
        if prev_waypoint is not None:
            # Apply partial reset: β̂ = α·β̂ + (1-α)·β̂₀
            self.beta_hat = self.alpha * self.beta_hat + (1 - self.alpha) * self.beta_hat0
            # Clamp after partial reset
            self.beta_hat = np.clip(self.beta_hat, -self.beta_max, self.beta_max)
