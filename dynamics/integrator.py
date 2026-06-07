"""RK4 数值积分器 + 全耦合动力学右端函数"""
import numpy as np
from config import I_SP, G0, T_MAX
from .gravity import gravity_accel


def thrust_accel(thrust, mass):
    """推力加速度矢量 a_thrust = T / m"""
    return thrust / max(mass, 1e-6)


def mass_flow_rate(thrust_mag):
    """质量变化率 ṁ = -|T| / (I_sp · g₀)，恒为负"""
    return -thrust_mag / (I_SP * G0)


def dynamics(t, state, thrust):
    """
    全耦合动力学右端函数  dx/dt = f(t, x, u)

    参数
    ----
    t : float
        当前时间 [s]（未使用，保留接口兼容性）
    state : np.ndarray, shape (7,)
        状态向量 [rx, ry, rz, vx, vy, vz, m]
    thrust : np.ndarray, shape (3,)
        推力矢量 [Tx, Ty, Tz] [N]，幅值在调用前已限幅

    返回
    ----
    dstate : np.ndarray, shape (7,)
        [vx, vy, vz, ax, ay, az, dm/dt]
    """
    r = state[0:3]
    v = state[3:6]
    m = state[6]
    thrust_mag = np.linalg.norm(thrust)
    a_grav = gravity_accel(r)
    a_thrust = thrust / max(m, 1e-6)
    dm = mass_flow_rate(thrust_mag)
    return np.array([
        v[0], v[1], v[2],
        a_grav[0] + a_thrust[0],
        a_grav[1] + a_thrust[1],
        a_grav[2] + a_thrust[2],
        dm
    ])


def rk4_step(f, t, state, thrust, dt):
    """
    四阶龙格-库塔积分一步

    参数
    ----
    f : callable
        右端函数 f(t, state, thrust) -> dstate
    t : float
        当前时间
    state : np.ndarray
        当前状态
    thrust : np.ndarray
        推力矢量（本步内视为常值）
    dt : float
        积分步长

    返回
    ----
    next_state : np.ndarray
        积分后的状态
    """
    k1 = f(t, state, thrust)
    k2 = f(t + dt * 0.5, state + dt * 0.5 * k1, thrust)
    k3 = f(t + dt * 0.5, state + dt * 0.5 * k2, thrust)
    k4 = f(t + dt, state + dt * k3, thrust)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
