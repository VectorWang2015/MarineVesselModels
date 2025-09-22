import numpy as np
from typing import Dict, Optional, List
from scipy.signal import savgol_filter


class LeastSquareFirstOrderNonLinearResponse():
    r"""
    Assumes that ship model conforms to:
        T \dot{r} + r + \alpha r^3 = K \delta
    """
    def __init__(
            self,
            time_step: float,
    ):
        self.time_step = time_step
        self.y = None   # y should be nx1
        self.H = None   # H should be nx3
        self.P = None   # P should be 3x3
        self.theta = None

    def _theta_to_params(
            self,
            theta: np.array,
    ) -> Dict[str, float]:
        a1 = theta[0][0]
        a2 = theta[1][0]
        b1 = theta[2][0]
        
        T = self.time_step / (1-a1)
        alpha = - a2 / (1-a1)
        K = b1 / (1-a1)

        return {"K": K, "T": T, "alpha": alpha}

    def identificate(
            self,
            train_data: np.array, # train data should be 7*n array
    ) -> Dict[str, float]:
        assert train_data.shape[0] == 7
        
        for i in range(train_data.shape[1] - 1):
        #for i in range(5):
            r = train_data[5][i]
            r_3 = r*r*r
            delta = train_data[6][i]
            r_next = train_data[5][i+1]
        
            y_i = np.array([r_next]).reshape([1, 1])
            H_i = np.array([r, r_3, delta]).reshape([1, 3])
        
            if self.y is None:
                self.y = y_i
                self.H = H_i
            else:
                self.y = np.vstack((self.y, y_i))
                self.H = np.vstack((self.H, H_i))
            
        self.P = np.linalg.inv(self.H.T @ self.H)
        self.theta = self.P @ self.H.T @ self.y
        return self._theta_to_params(theta=self.theta)


class RecursiveLeastSquareFirstOrderNonLinearResponse(
    LeastSquareFirstOrderNonLinearResponse
):
    r"""
    Assumes that ship model conforms to:
        T \dot{r} + r + \alpha r^3 = K \delta
    """
    def __init__(
            self,
            time_step: float,
    ):
        super().__init__(time_step=time_step)
        self.K = None

    def recursive_identificate_once(
            self,
            H_k: np.array,
            y_k: np.array,
    ):
        # update K
        self.K = self.P @ H_k.T @ np.linalg.inv(np.identity(1) + H_k @ self.P @ H_k.T)
        # update P
        self.P = (np.identity(3) - self.K @ H_k) @ self.P
        # update theta
        self.theta = self.theta + self.K @ (y_k - H_k@self.theta)

    def recursive_identificate(self, train_data: np.array, thetas: Optional[List[np.array]]=None) -> Dict[str, float]:
        """
        thetas is used to track convergence of self.theta
        """
        # clear and reinit H, y
        self.y = None   # nx1
        self.H = None   # nx3

        for i in range(train_data.shape[1] - 1):
        #for i in range(5):
            r = train_data[5][i]
            r_3 = r*r*r
            delta = train_data[6][i]
            r_next = train_data[5][i+1]
        
            y_i = np.array([r_next]).reshape([1, 1])
            H_i = np.array([r, r_3, delta]).reshape([1, 3])
        
            if self.y is None:
                self.y = y_i
                self.H = H_i
            else:
                self.y = np.vstack((self.y, y_i))
                self.H = np.vstack((self.H, H_i))

        if thetas is not None:
            thetas.append(self.theta)

        for i in range(self.H.shape[0]):
            self.recursive_identificate_once(H_k=self.H[i:i+1,:], y_k=self.y[i:i+1,:])
            if thetas is not None:
                thetas.append(self.theta)

        return self._theta_to_params(theta=self.theta)

    def identificate(self, train_data: np.array,thetas: Optional[List[np.array]]=None) -> Dict[str, float]:
        # if first time compute, use non-recursive method
        if self.P is None:
            return super().identificate(train_data)

        # recursive update
        return self.recursive_identificate(train_data, thetas)


class LeastSquareFossen():
    r"""
        y = tau

        \theta = \begin{bmatrix}
        m_{11} \\
        m_{22} \\
        m_{33} \\
        d_{11} \\
        d_{22} \\
        d_{33} \\
        \end{bmatrix} , 

        H = \begin{bmatrix}
        \dot{u} & -vr & 0 & u & 0 & 0 \\
        ur & \dot{v} & 0 & 0 & v & 0 \\
        uv & -uv & \dot{r} & 0 & 0 & r
        \end{bmatrix}

        !!! neglect \tau_2 is unworkable,
            removing \tau_2 (y_2) will cause H.T @ H to be singular
    """
    def __init__(
            self,
            time_step: float,
    ):
        self.time_step = time_step
        self.y = None   # y should be nx1
        self.H = None   # H should be nx6
        self.P = None   # P should be 6x6
        self.theta = None

    def _theta_to_params(
            self,
            theta: np.array,
    ) -> Dict[str, float]:
        m11 = theta[0][0]
        m22 = theta[1][0]
        m33 = theta[2][0]
        d11 = theta[3][0]
        d22 = theta[4][0]
        d33 = theta[5][0]
        
        return {
            "m11": m11, "m22": m22, "m33": m33,
            "d11": d11, "d22": d22, "d33": d33,
        }

    def smooth_fn(self, signals):
        """
        Filter ignored here, cuz this class is inherited by RLS
        """
        return signals

    def dot_fn(self, signals):
        """
        Filter ignored here, cuz this class is inherited by RLS
        """
        return np.gradient(signals, self.time_step)

    def norm_fn(self, H):
        """
        Normalization ignored here, cuz this class is inherited by RLS
        """
        return H

    def denorm_fn(self, theta):
        """
        Normalization ignored here, cuz this class is inherited by RLS
        """
        return theta

    def identificate(
            self,
            train_data: np.array, # train data should be 9*n array
    ) -> Dict[str, float]:
        assert train_data.shape[0] == 9

        us = self.smooth_fn(train_data[3])
        vs = self.smooth_fn(train_data[4])
        rs = self.smooth_fn(train_data[5])
        # us = train_data[3]
        # vs = train_data[4]
        # rs = train_data[5]

        dot_us = self.dot_fn(us)
        dot_vs = self.dot_fn(vs)
        dot_rs = self.dot_fn(rs)
        
        for i in range(train_data.shape[1] - 1):
        #for i in range(5):
            u = us[i]
            v = vs[i]
            r = rs[i]
            dot_u = dot_us[i]
            dot_v = dot_vs[i]
            dot_r = dot_rs[i]

            # construct tau & H
            y_i = train_data[6:,i].reshape([3, 1])
            H_i = np.array(
                [
                    [dot_u, -v*r, 0, u, 0, 0],
                    [u*r, dot_v, 0, 0, v, 0],
                    [u*v, -u*v, dot_r, 0, 0, r],

                ]
            )
        
            if self.y is None:
                self.y = y_i
                self.H = H_i
            else:
                self.y = np.vstack((self.y, y_i))
                self.H = np.vstack((self.H, H_i))

        self.H = self.norm_fn(self.H)
        self.P = np.linalg.inv(self.H.T @ self.H)
        self.theta = self.P @ self.H.T @ self.y
        self.theta = self.denorm_fn(self.theta)

        return self._theta_to_params(theta=self.theta)


class LeastSquareFossenSG(LeastSquareFossen):
    def __init__(
            self,
            time_step: float,
            SG_window: int = 11,
    ):
        """
        Savitzky–Golay Filtered LS Method for Fossen models
        recommend SG window size 11~21 for dt=0.1, 51~101 for dt=0.01
        """
        super().__init__(time_step)
        self.SG_window = SG_window

    def smooth_fn(self, signals):
        return savgol_filter(signals, window_length=self.SG_window, polyorder=3)

    def dot_fn(self, signals):
        return savgol_filter(signals, window_length=self.SG_window, polyorder=3, deriv=1, delta=self.time_step)

    def norm_fn(self, H):
        self.scale = np.linalg.norm(H, axis=0)
        print(f"Normalization imposed, scale: {self.scale}")
        return H / self.scale

    def denorm_fn(self, theta):
        assert hasattr(self, "scale")
        print(f"Denormalization imposed, scale: {self.scale}")
        return theta / self.scale


class RecursiveLeastSquareFossen(
    LeastSquareFossen
):
    r"""
    !!! need to implement filter to denoise and better fit dot u
        y = tau

        \theta = \begin{bmatrix}
        m_{11} \\
        m_{22} \\
        m_{33} \\
        d_{11} \\
        d_{22} \\
        d_{33} \\
        \end{bmatrix} , 

        H = \begin{bmatrix}
        \dot{u} & -vr & 0 & u & 0 & 0 \\
        ur & \dot{v} & 0 & 0 & v & 0 \\
        uv & -uv & \dot{r} & 0 & 0 & r
        \end{bmatrix}
    """
    def __init__(
            self,
            time_step: float,
    ):
        super().__init__(time_step=time_step)
        self.K = None

    def recursive_identificate_once(
            self,
            H_k: np.array,
            y_k: np.array,
    ):
        # update K
        self.K = self.P @ H_k.T @ np.linalg.inv(np.identity(1) + H_k @ self.P @ H_k.T)
        # update P
        self.P = (np.identity(6) - self.K @ H_k) @ self.P
        # update theta
        self.theta = self.theta + self.K @ (y_k - H_k@self.theta)

    def recursive_identificate(self, train_data: np.array, thetas: Optional[List[np.array]]=None) -> Dict[str, float]:
        """
        thetas is used to track convergence of self.theta
        """
        # clear and reinit H, y
        self.y = None   # nx1
        self.H = None   # nx6

        us = train_data[3]
        vs = train_data[4]
        rs = train_data[5]

        dot_us = np.gradient(us, self.time_step)
        dot_vs = np.gradient(vs, self.time_step)
        dot_rs = np.gradient(rs, self.time_step)
        
        for i in range(train_data.shape[1] - 1):
        #for i in range(5):
            u = us[i]
            v = vs[i]
            r = rs[i]
            dot_u = dot_us[i]
            dot_v = dot_vs[i]
            dot_r = dot_rs[i]

            # construct tau & H
            y_i = train_data[6:,i].reshape([3, 1])
            H_i = np.array(
                [
                    [dot_u, -v*r, 0, u, 0, 0],
                    [u*r, dot_v, 0, 0, v, 0],
                    [u*v, -u*v, dot_r, 0, 0, r],

                ]
            )
        
            if self.y is None:
                self.y = y_i
                self.H = H_i
            else:
                self.y = np.vstack((self.y, y_i))
                self.H = np.vstack((self.H, H_i))

        if thetas is not None:
            thetas.append(self.theta)

        for i in range(self.H.shape[0]):
            self.recursive_identificate_once(H_k=self.H[i:i+1,:], y_k=self.y[i:i+1,:])
            if thetas is not None:
                thetas.append(self.theta)

        return self._theta_to_params(theta=self.theta)

    def identificate(self, train_data: np.array,thetas: Optional[List[np.array]]=None) -> Dict[str, float]:
        # if first time compute, use non-recursive method
        if self.P is None:
            return super().identificate(train_data)

        # recursive update
        return self.recursive_identificate(train_data, thetas)


class AlternatingLeastSquareFossen():
    r"""
        y = tau

        \theta = \begin{bmatrix}
        m_{11} \\
        m_{22} \\
        m_{33} \\
        d_{11} \\
        d_{22} \\
        d_{33} \\
        \end{bmatrix} , 

        H = \begin{bmatrix}
        \dot{u} & -vr & 0 & u & 0 & 0 \\
        ur & \dot{v} & 0 & 0 & v & 0 \\
        uv & -uv & \dot{r} & 0 & 0 & r
        \end{bmatrix}

        F = cvn + d|n|n
        tau1 = Fl + Fr = (nl+nr) v c + (|nl|nl + |nr|nr) d
        tau2 = 0
        tau3 = b/2 * (Fl - Fr) = (b/2) v (nl - nr) c + (b/2) (|nl|nl - |nr|nr) d
    """
    def __init__(
            self,
            time_step: float,
            b: float,
            init_c = -1.6e-4,
            init_d = 5.04e-3,
            threshold: float=1e-4,
    ):
        self.time_step = time_step
        self.b = b
        self.y = None   # y should be nx1
        self.H = None   # H should be nx6
        self.P = None   # P should be 6x6
        self.theta = None

        self.y_cd = None
        self.H_cd = None
        self.P_cd = None
        self.thrust_params = np.array([init_c, init_d]).reshape([2, 1])

        self.threshold = threshold


    def _theta_to_params(
            self,
    ) -> Dict[str, float]:
        m11 = self.theta[0][0]
        m22 = self.theta[1][0]
        m33 = self.theta[2][0]
        d11 = self.theta[3][0]
        d22 = self.theta[4][0]
        d33 = self.theta[5][0]
        c = self.thrust_params[0][0]
        d = self.thrust_params[1][0]
        
        return {
            "m11": m11, "m22": m22, "m33": m33,
            "d11": d11, "d22": d22, "d33": d33,
            "c": c, "d": d,
        }

    def construct_y_MD(
            self,
            train_data: np.array, # train data should be 8*n array
    ):
        # update y each time
        self.y = None

        us = train_data[3]
        vs = train_data[4]
        n_ls = train_data[6]
        n_rs = train_data[7]

        c = self.thrust_params[0][0]
        d = self.thrust_params[1][0]

        for i in range(train_data.shape[1] - 1):
            u = us[i]
            #v = vs[i]
            #V = np.sqrt(u**2+v**2)
            # v is small, V = u is a better approximation
            V = u

            n_l = n_ls[i]
            n_r = n_rs[i]

            F_l = c*V*n_l + d*abs(n_l)*n_l
            F_r = c*V*n_r + d*abs(n_r)*n_r

            tau_1 = F_l + F_r
            tau_2 = 0
            tau_3 = (F_l - F_r) * self.b / 2

            y_i = np.array([tau_1, tau_2, tau_3]).reshape([3, 1])

            if self.y is None:
                self.y = y_i
            else:
                self.y = np.vstack((self.y, y_i))

    def construct_H_MD(
            self,
            train_data: np.array, # train data should be 8*n array
    ):
        self.H = None
        us = train_data[3]
        vs = train_data[4]
        rs = train_data[5]

        dot_us = np.gradient(us, self.time_step)
        dot_vs = np.gradient(vs, self.time_step)
        dot_rs = np.gradient(rs, self.time_step)
        
        for i in range(train_data.shape[1] - 1):
        #for i in range(5):
            u = us[i]
            v = vs[i]
            r = rs[i]
            dot_u = dot_us[i]
            dot_v = dot_vs[i]
            dot_r = dot_rs[i]
            # v is small, V = u is a better approximation
            #V = np.sqrt(u**2+v**2)
            V = u

            # construct tau & H
            H_i = np.array(
                [
                    [dot_u, -v*r, 0, u, 0, 0],
                    [u*r, dot_v, 0, 0, v, 0],
                    [u*v, -u*v, dot_r, 0, 0, r],

                ]
            )
        
            if self.H is None:
                self.H = H_i
            else:
                self.H = np.vstack((self.H, H_i))

    def identificate_MD(self) -> bool:
        """
        returns if the identified {M, D} meets threshold requirements
        """
        # self.P = np.linalg.inv(self.H.T @ self.H)
        # theta = self.P @ self.H.T @ self.y
        theta, *_ = np.linalg.lstsq(self.H, self.y, rcond=None)

        # first time
        if self.theta is None:
            self.theta = theta
            return False
        
        # check residual and returns if erros within threshold
        residual = np.max(np.abs(theta - self.theta))
        #print(f"Current average MD residual: {residual}")

        self.theta = theta
        #print(self.theta)
        return residual <= self.threshold

    def construct_y_cd(
            self,
            train_data: np.array, # train data should be 8*n array
    ):
        """
        tau_2 = 0, therefore neglected
        """
        self.y_cd = None

        us = train_data[3]
        vs = train_data[4]
        rs = train_data[5]

        dot_us = np.gradient(us, self.time_step)
        dot_vs = np.gradient(vs, self.time_step)
        dot_rs = np.gradient(rs, self.time_step)

        # extract M, D params
        m11 = self.theta[0][0]
        m22 = self.theta[1][0]
        m33 = self.theta[2][0]
        d11 = self.theta[3][0]
        d22 = self.theta[4][0]
        d33 = self.theta[5][0]
        c = self.thrust_params[0][0]
        d = self.thrust_params[1][0]
        
        for i in range(train_data.shape[1] - 1):
            u = us[i]
            v = vs[i]
            r = rs[i]
            dot_u = dot_us[i]
            dot_v = dot_vs[i]
            dot_r = dot_rs[i]

            y_1 = m11*dot_u + d11*u - m22*v*r
            # y_2 should be zero since tau_2 == 0, therefore neglected
            y_3 = m33*dot_r + d33*r + m22*v*u - m11*v*u

            y_i = np.array([y_1, y_3]).reshape([2, 1])

            if self.y_cd is None:
                self.y_cd = y_i
            else:
                self.y_cd = np.vstack((self.y_cd, y_i))

    def construct_H_cd(
            self,
            train_data: np.array, # train data should be 8*n array
    ):
        """
        tau_2 = 0, therefore neglected
        """
        self.H_cd = None

        us = train_data[3]
        vs = train_data[4]
        rs = train_data[5]

        n_ls = train_data[6]
        n_rs = train_data[7]

        dot_us = np.gradient(us, self.time_step)
        dot_vs = np.gradient(vs, self.time_step)
        dot_rs = np.gradient(rs, self.time_step)

        for i in range(train_data.shape[1] - 1):
            u = us[i]
            v = vs[i]
            r = rs[i]
            dot_u = dot_us[i]
            dot_v = dot_vs[i]
            dot_r = dot_rs[i]

            n_l = n_ls[i]
            n_r = n_rs[i]
            
            # v is small, V = u is a better approximation
            #V = np.sqrt(u**2+v**2)
            V = u

            H_11 = V * (n_l+n_r)
            H_12 = abs(n_l)*n_l + abs(n_r)*n_r
            H_21 = self.b * V * (n_l - n_r) / 2
            H_22 = self.b * (abs(n_l)*n_l - abs(n_r)*n_r) / 2

            H_i = np.array([[H_11, H_12], [H_21, H_22]])

            if self.H_cd is None:
                self.H_cd = H_i
            else:
                self.H_cd = np.vstack((self.H_cd, H_i))

    def identificate_cd(
            self,
    ) -> bool:
        """
        returns if the identified {c, d} meets threshold requirements
        """
        # self.P_cd = np.linalg.inv(self.H_cd.T @ self.H_cd)
        # thrust_params = self.P_cd @ self.H_cd.T @ self.y_cd
        thrust_params, *_ = np.linalg.lstsq(self.H_cd, self.y_cd, rcond=None)

        # check residual and returns if erros within threshold
        residual = np.max(np.abs(self.thrust_params - thrust_params))
        #print(f"Current average cd residual: {residual}")

        self.thrust_params = thrust_params
        return residual <= self.threshold

    def identificate(
            self,
            train_data: np.array, # train data should be 8*n array
    ) -> Dict[str, float]:
        assert train_data.shape[0] == 8

        MD_is_ok = False
        cd_is_ok = False

        # for {M, D} identification, H is fixed and based on train data
        # y is tau, which depends on both {c, d} and train data
        # construct H for {M, D} indentificate once
        self.construct_H_MD(train_data)

        # recursively update {M, D} & {c, d} until meet threshold
        while not MD_is_ok or not cd_is_ok:
            # construct y for {M, D}
            self.construct_y_MD(train_data)
            # update {M, D}
            MD_is_ok = self.identificate_MD()
            # construct H, y for {c, d}
            self.construct_H_cd(train_data)
            self.construct_y_cd(train_data)
            cd_is_ok = self.identificate_cd()

        return self._theta_to_params()
