import numpy as np
from typing import List
from .los import LOSGuider
from .pid import DoubleLoopHeadingPID, PIDAW


class DualThrustUSVStopAndAdjustController():
    """
    this controller takes the strategy of:
        while psi error exceeds threshold: zero-radius turn with target u == 0
        while psi error whithin threshold: crusing with rareget u == tgt_u
    """
    def __init__(
            self,
            psi_threshold: float,
            u_cruise: float,
            psi_controller: DoubleLoopHeadingPID,
            u_controller: PIDAW,
            guider: LOSGuider,
    ):
        self.psi_threshold = psi_threshold
        self.u_cruise = u_cruise
        self.psi_controller = psi_controller
        self.u_controller = u_controller
        self.guider = guider
        
        # a list of states, each state a nx1 ndarray
        self.states: List[np.ndarray] = []
        self.desired_psis: List[float] = []
        self.desired_us: List[float] = []
        self.desired_rs: List[float] = []
        self.left_thru: List[float] = []
        self.right_thru: List[float] = []

    def step(
            self,
            state: np.ndarray,
    ):
        """
        takes in a state, returns None if reached target,
        else returns (left_thrust, right_thrust) in Newton
        """
        x = state[0][0]
        y = state[1][0]
        psi = state[2][0]
        u = state[3][0]
        v = state[4][0]
        r = state[5][0]

        is_ended, desired_psi = self.guider.step((x, y), psi)
        if is_ended:
            # returns None indicates reached final target
            return None

        # if not ended yet
        psi_err = ((desired_psi - psi) + np.pi) % (2*np.pi) - np.pi

        if abs(psi_err) >= self.psi_threshold:
            desired_u = 0
        else:
            desired_u = self.u_cruise

        diff_control_signal, ref_r = self.psi_controller.step(psi_ref=desired_psi, psi=psi, r=r)
        velo_control_signal = self.u_controller.step(sp=desired_u, y=u)

        left = velo_control_signal + diff_control_signal
        right = velo_control_signal - diff_control_signal

        # record
        self.states.append(state)
        self.desired_psis.append(desired_psi)
        self.desired_us.append(desired_u)
        self.desired_rs.append(ref_r)
        self.left_thru.append(left)
        self.right_thru.append(right)

        return (left, right)


    def summarize(self):
        states = np.hstack(self.states)
        return {
            "xs": states[0, :],
            "ys": states[1, :],
            "psis": states[2, :],
            "us": states[3, :],
            "vs": states[4, :],
            "rs": states[5, :],
            "lefts": np.array(self.left_thru),
            "rights": np.array(self.right_thru),
            "desired_psis": np.array(self.desired_psis),
            "desired_us": np.array(self.desired_us),
            "desired_rs": np.array(self.desired_rs),
        }