import numpy as np


sample_thrust = 50.0
sample_b = 0.52
sample_hydro_params = {
    "m11": 50.05,
    "m22": 84.36,
    "m33": 17.21,
    "Xu": 151.57,
    "Yv": 132.5,
    "Nr": 34.56,
    "c": -1.60e-4,
    "d": 5.04e-3,
}


class MMG_HXH:
    def __init__(
            self,
            m11,
            m22,
            m33,
            Xu,
            Yv,
            Nr,
            c,
            d,
    ):
        """初始化差分无人水面艇模型参数"""
        # 模型参数（来源于论文中的系统识别结果）
        self.m11 = m11
        self.m22 = m22
        self.m33 = m33
        self.Xu = Xu
        self.Yv = Yv
        self.Nr = Nr
        self.c = c
        self.d = d

    def partial_state(self, state, tau):
        """使用离散时间模型更新状态"""
        u, v, r, psi = state[3][0], state[4][0], state[5][0], state[2][0]

        # 控制输入
        tau_u = tau[0][0]
        tau_r = tau[2][0]

        # 动力学方程
        u_dot = (1 / self.m11) * (tau_u + self.m22 * v * r - self.Xu * u)  # 前进加速度
        v_dot = (1 / self.m22) * (-self.m11 * u * r - self.Yv * v)         # 横移加速度
        r_dot = (1 / self.m33) * (tau_r - (self.m22 - self.m11) * u * v - self.Nr * r)  # 偏航角加速度
        dot_v = np.array([float(u_dot), float(v_dot), float(r_dot)]).reshape((3, 1))  # 状态导数

        dot_x = np.array([
            np.cos(psi)*u - np.sin(psi)*v,
            np.sin(psi)*u + np.cos(psi)*v,
            r,
        ]).reshape([3, 1])
        # 运动学更新
        return np.vstack((dot_x, dot_v))
