import numpy as np
from typing import Dict
from scipy.signal import savgol_filter

from MarineVesselModels.thrusters import NpsDoubleThruster


class WeightedRidgeLSFossen:
    r"""
    Improvements:
      - SG smooth/differential
      - column normalization
      - WLS
      - Ridge regularization
    """

    def __init__(
        self,
        time_step: float,
        SG_window: int = 11,
        weights=(1.0, 1, 2.5),   # (Fx, Fy, Nr) weight
        lam: float = 1e-3,          # ridge regularization weight
        enable_filter: bool = True,
    ):
        self.time_step = time_step
        self.SG_window = SG_window
        self.weights = weights
        self.lam = lam
        self.scale = None
        self.theta = None

        self.enable_filter = enable_filter

    def _theta_to_params(self, theta: np.array) -> Dict[str, float]:
        return {
            "m11": theta[0, 0],
            "m22": theta[1, 0],
            "m33": theta[2, 0],
            "d11": theta[3, 0],
            "d22": theta[4, 0],
            "d33": theta[5, 0],
        }

    def smooth_fn(self, signals):
        return savgol_filter(signals, window_length=self.SG_window, polyorder=3)

    def dot_fn(self, signals):
        return savgol_filter(
            signals, window_length=self.SG_window,
            polyorder=3, deriv=1, delta=self.time_step
        )

    def construct_matrices(
            self,
            us: np.array,
            vs: np.array,
            rs: np.array,
            dot_us: np.array,
            dot_vs: np.array,
            dot_rs: np.array,
            taus: np.array,
    ) -> Dict[str, float]:

        if self.enable_filter:
            us = self.smooth_fn(us)
            vs = self.smooth_fn(vs)
            rs = self.smooth_fn(rs)
    
            dot_us = self.smooth_fn(dot_us)
            dot_vs = self.smooth_fn(dot_vs)
            dot_rs = self.smooth_fn(dot_rs)

        Hs, Ys = [], []
        for i in range(us.shape[0] - 1):
            u, v, r = us[i], vs[i], rs[i]
            dot_u, dot_v, dot_r = dot_us[i], dot_vs[i], dot_rs[i]
            tau = taus[:,i].reshape([3, 1])

            H_i = np.array([
                [dot_u, -v*r, 0, u, 0, 0],
                [u*r, dot_v, 0, 0, v, 0],
                [-u*v, u*v, dot_r, 0, 0, r],
            ])

            Hs.append(H_i)
            Ys.append(tau)

        H = np.vstack(Hs)
        y = np.vstack(Ys)

        self.H = H
        self.y = y

        return H, y

    def identify(
            self,
            us: np.array,
            vs: np.array,
            rs: np.array,
            dot_us: np.array,
            dot_vs: np.array,
            dot_rs: np.array,
            taus: np.array,
    ) -> Dict[str, float]:
        H, y = self.construct_matrices(us, vs, rs, dot_us, dot_vs, dot_rs, taus)

        # column norm, 1e-12 for perventing 0
        self.scale = np.linalg.norm(H, axis=0) + 1e-12
        Hs = H / self.scale

        # channel weight
        w_x, w_y, w_r = self.weights
        N = us.shape[0] - 1
        W = np.diag([w_x, w_y, w_r]*N)

        # ridge regularization
        A = Hs.T @ W @ Hs + self.lam * np.eye(Hs.shape[1])
        b = Hs.T @ W @ y
        # linalg.solve soves a well-determined full-rank, linear matrix equation Ax=b
        theta_s = np.linalg.solve(A, b)

        # denorm
        self.theta = theta_s / self.scale.reshape(-1, 1)

        return self._theta_to_params(self.theta)
    
    def compute_residuals(self) -> Dict[str, float]:
        """
        returns residuals for each channel: Fx, Fy, Nr
        """
        if self.theta is None or self.H is None or self.y is None:
            raise RuntimeError("call identify() first!")

        y_pred = self.H @ self.theta
        residuals = self.y - y_pred

        res_fx = residuals[0::3]
        res_fy = residuals[1::3]
        res_nr = residuals[2::3]

        rms_fx = np.sqrt(np.mean(res_fx**2))
        rms_fy = np.sqrt(np.mean(res_fy**2))
        rms_nr = np.sqrt(np.mean(res_nr**2))

        return {"Fx_RMS": rms_fx, "Fy_RMS": rms_fy, "Nr_RMS": rms_nr}



class WeightedRidgeLSThruster:
    r"""
    Assume:
        F = cun + d|n|n
    Improvements:
      - SG smooth/differential
      - column normalization
      - WLS
      - Ridge regularization
    """

    def __init__(
        self,
        time_step: float,
        b: float,
        SG_window: int = 11,
        weights=(1.0, 2.5),   # (Fx, Fy, Nr) weight
        lam: float = 1e-3,          # ridge regularization weight
        enable_filter: bool = True,
    ):
        self.b = b  # width, distance between two thrusters
        self.time_step = time_step
        self.SG_window = SG_window
        self.weights = weights
        self.lam = lam
        self.scale = None
        self.theta = None

        self.enable_filter = enable_filter

    def set_hydro_params(
        self,
        m11: float,
        m22: float,
        m33: float,
        d11: float,
        d22: float,
        d33: float,
    ):
        self.m11 = m11
        self.m22 = m22
        self.m33 = m33
        self.d11 = d11
        self.d22 = d22
        self.d33 = d33

    def _theta_to_params(self, theta: np.array) -> Dict[str, float]:
        return {
            "c": theta[0, 0],
            "d": theta[1, 0],
        }

    def smooth_fn(self, signals):
        return savgol_filter(signals, window_length=self.SG_window, polyorder=3)

    def dot_fn(self, signals):
        return savgol_filter(
            signals, window_length=self.SG_window,
            polyorder=3, deriv=1, delta=self.time_step
        )

    def construct_matrices(
            self,
            us: np.array,
            vs: np.array,
            rs: np.array,
            dot_us: np.array,
            dot_vs: np.array,
            dot_rs: np.array,
            rps_ls: np.array,
            rps_rs: np.array,
    ) -> Dict[str, float]:
        # hydro params should exist before this func is called
        assert hasattr(self, "m11")

        if self.enable_filter:
            us = self.smooth_fn(us)
            vs = self.smooth_fn(vs)
            rs = self.smooth_fn(rs)
    
            dot_us = self.smooth_fn(dot_us)
            dot_vs = self.smooth_fn(dot_vs)
            dot_rs = self.smooth_fn(dot_rs)

        Hs, Ys = [], []
        for i in range(us.shape[0] - 1):
            u, v, r = us[i], vs[i], rs[i]
            dot_u, _, dot_r = dot_us[i], dot_vs[i], dot_rs[i]
            rps_l, rps_r = rps_ls[i], rps_rs[i]

            tau_i = np.array([
                [self.m11*dot_u - self.m22*v*r + self.d11*u],
                [self.m33*dot_r + self.m22*u*v - self.m11*u*v + self.d33*r],
            ])

            H_i = np.array([
                [u*(rps_l + rps_r), abs(rps_l)*rps_l + abs(rps_r)*rps_r],
                [u*(self.b/2)*(rps_l - rps_r), (self.b/2)*(abs(rps_l)*rps_l - abs(rps_r)*rps_r)],  # 注意符号
            ])

            Hs.append(H_i)
            Ys.append(tau_i)

        H = np.vstack(Hs)
        y = np.vstack(Ys)

        self.H = H
        self.y = y

        return H, y

    def identify(
            self,
            us: np.array,
            vs: np.array,
            rs: np.array,
            dot_us: np.array,
            dot_vs: np.array,
            dot_rs: np.array,
            rps_ls: np.array,
            rps_rs: np.array,
    ) -> Dict[str, float]:
        """
        """
        H, y = self.construct_matrices(us, vs, rs, dot_us, dot_vs, dot_rs, rps_ls, rps_rs)

        # column norm, 1e-12 for perventing 0
        self.scale = np.linalg.norm(H, axis=0) + 1e-12
        Hs = H / self.scale

        # channel weight
        w_x, w_r = self.weights
        N = rps_ls.shape[0]
        W = np.diag([w_x, w_r]*N)

        # ridge regularization
        A = Hs.T @ W @ Hs + self.lam * np.eye(Hs.shape[1])
        b = Hs.T @ W @ y
        # linalg.solve soves a well-determined full-rank, linear matrix equation Ax=b
        theta_s = np.linalg.solve(A, b)

        # denorm
        self.theta = theta_s / self.scale.reshape(-1, 1)

        return self._theta_to_params(self.theta)
    
    def compute_residuals(self) -> Dict[str, float]:
        """
        returns residuals for each channel: Fx, Nr
        """
        if self.theta is None or self.H is None or self.y is None:
            raise RuntimeError("call identify() first!")

        y_pred = self.H @ self.theta
        residuals = self.y - y_pred

        res_fx = residuals[0::2]
        res_nr = residuals[1::2]
       
        rms_fx = np.sqrt(np.mean(res_fx**2))
        rms_nr = np.sqrt(np.mean(res_nr**2))

        return {"Fx_RMS": rms_fx, "Nr_RMS": rms_nr}


class AltLS():
    def __init__(
            self,
            time_step: float,
            b: float,
            init_thru_params: Dict,
            hull_ls_params: Dict,
            thru_ls_params: Dict,
    ):
        self.hull_ls = WeightedRidgeLSFossen(
            time_step=time_step, **hull_ls_params,
        )
        self.thru_ls = WeightedRidgeLSThruster(
            time_step=time_step, b=b, **thru_ls_params,
        )

        self.current_hull_params = None
        self.current_thru_params = init_thru_params.copy()

        self.thruster = NpsDoubleThruster(b=b, **init_thru_params)

        self.current_result = None

    def calcul_result_relative_difference(
        self,
        prev_result: Dict,
        curr_result: Dict,
    ) -> Dict:
        result_diff = {}
        for key in prev_result:
            prev_v = prev_result[key]
            curr_v = curr_result[key]
            if prev_v != 0:
                relative_diff = (curr_v - prev_v) / prev_v
            else:
                relative_diff = None
            
            result_diff[key] = relative_diff
        return result_diff
            
    
    def rps_to_taus(
            self,
            us: np.array,
            rps_ls: np.array,
            rps_rs: np.array,
    ) -> np.array:
        taus = []
        self.thruster.update_params(**self.current_thru_params)
        for u, l, r in zip(us, rps_ls, rps_rs):
            tau = self.thruster.n_to_tau(l_nps=float(l), r_nps=float(r), u=float(u))
            taus.append(tau)
        return np.hstack(taus)

    def identify_once(
            self,
            us: np.array,
            vs: np.array,
            rs: np.array,
            dot_us: np.array,
            dot_vs: np.array,
            dot_rs: np.array,
            rps_ls: np.array,
            rps_rs: np.array,
    ) -> Dict[str, float]:
        # generate taus through current thru params
        taus = self.rps_to_taus(us=us, rps_ls=rps_ls, rps_rs=rps_rs)
        # identify hull params with current thru params
        self.current_hull_params = self.hull_ls.identify(
            us,
            vs,
            rs,
            dot_us,
            dot_vs,
            dot_rs,
            taus,
        )
        # identify thru params with current hull params
        self.thru_ls.set_hydro_params(**self.current_hull_params)
        self.current_thru_params = self.thru_ls.identify(
            us,
            vs,
            rs,
            dot_us,
            dot_vs,
            dot_rs,
            rps_ls,
            rps_rs,
        )
        # calcul relative changes
        if not self.current_result:
            self.current_result = {**self.current_hull_params, **self.current_thru_params}
            return self.current_result, None
        else:
            prev_result = self.current_result
            self.current_result = {**self.current_hull_params, **self.current_thru_params}
            relative_difference = self.calcul_result_relative_difference(
                prev_result,
                self.current_result,
            )
            return self.current_result, relative_difference