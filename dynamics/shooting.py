"""数值打靶法 — scipy.optimize 求解终端约束 + 燃料优化

控制参数化: 将推力剖面用 N 段等时长推力值表示
求解器: L-BFGS-B, 目标 = 燃料 + 终端约束罚函数
"""
import numpy as np
from scipy.optimize import minimize
from config import T_MAX, I_SP, G0, DT, MU


def _gravity_1d(dist_from_center):
    r = max(dist_from_center, 1.0)
    return -MU / (r * r)


def _simulate(params, d0, v0, m0, n_seg, total_time):
    """前向仿真，返回 (d_final, v_final, fuel, trajectory)"""
    d, v, m, fuel = float(d0), float(v0), float(m0), 0.0
    seg_dt = total_time / n_seg
    traj = []
    for i in range(n_seg):
        u = float(np.clip(params[i], -T_MAX, T_MAX))
        steps = int(seg_dt / DT)
        for _ in range(steps):
            a_t = u / max(m, 1e-6)
            a_g = _gravity_1d(d)
            v += (a_t + a_g) * DT
            d = max(d - v * DT, 0.0)
            dm = -abs(u) * DT / (I_SP * G0)
            m = max(m + dm, 1.0)
            fuel += abs(u) * DT / (I_SP * G0)
            traj.append((d, v, m, u))
            if abs(d) < 0.01:
                break
        if abs(d) < 0.01:
            break
    return float(d), float(v), float(fuel), traj


def shooting_solve(r0, v0, r_target, m0, n_seg=8, total_time=800.0):
    """
    打靶法求解可行轨迹 (满足终端约束，不追求燃料最优)

    返回 3D 参考轨迹 list[np.array(7,)]
    """
    delta = r_target - r0
    d0 = np.linalg.norm(delta)
    unit = delta / max(d0, 1e-6)
    v_radial = float(np.dot(v0, unit))

    def err(params):
        df, vf, _, _ = _simulate(params, d0, v_radial, m0, n_seg, total_time)
        return df**2 + vf**2

    x0 = np.zeros(n_seg)
    x0[-2:] = -T_MAX * 0.5
    bounds = [(-T_MAX, T_MAX)] * n_seg
    res = minimize(err, x0=x0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 150})

    df, vf, fuel, rad_traj = _simulate(res.x, d0, v_radial, m0, n_seg, total_time)

    ref, t = [], 0.0
    for d, vr, m_t, u in rad_traj:
        r = r_target - unit * d
        v = -unit * vr
        ref.append(np.array([*r, *v, m0, t]))
        t += DT
    ref.append(np.array([*r_target, 0.0, 0.0, 0.0, m0, t]))
    return ref


def shooting_optimize_fuel(r0, v0, r_target, m0, n_seg=10, total_time=800.0):
    """
    打靶法 + 燃料优化: minimize fuel subject to terminal constraints

    返回 (ref_traj, fuel_used, d_final, v_final)
    """
    delta = r_target - r0
    d0 = np.linalg.norm(delta)
    unit = delta / max(d0, 1e-6)
    v_radial = float(np.dot(v0, unit))

    def objective(params):
        df, vf, fuel, _ = _simulate(params, d0, v_radial, m0, n_seg, total_time)
        return fuel + 500.0 * (df**2 + vf**2)

    x0 = np.zeros(n_seg)
    bounds = [(-T_MAX, T_MAX)] * n_seg
    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 200})

    df, vf, fuel, rad_traj = _simulate(res.x, d0, v_radial, m0, n_seg, total_time)

    ref, t = [], 0.0
    for d, vr, m_t, u in rad_traj:
        r = r_target - unit * d
        v = -unit * vr
        ref.append(np.array([*r, *v, m0, t]))
        t += DT
    ref.append(np.array([*r_target, 0.0, 0.0, 0.0, m0, t]))
    return ref, fuel, float(df), float(vf)


def get_ref_at_step(ref_traj, step):
    return ref_traj[step] if step < len(ref_traj) else ref_traj[-1]
