import numpy as np
from typing import Dict, Optional, List


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

    def identificate(
            self,
            train_data: np.array, # train data should be 9*n array
    ) -> Dict[str, float]:
        assert train_data.shape[0] == 9

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
            
        self.P = np.linalg.inv(self.H.T @ self.H)
        self.theta = self.P @ self.H.T @ self.y

        return self._theta_to_params(theta=self.theta)


class RecursiveLeastSquareFossen(
    LeastSquareFossen
):
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
