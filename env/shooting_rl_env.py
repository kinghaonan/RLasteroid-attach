"""
打靶法参考轨迹 + RL 1D 修正 —— 燃料优化环境

架构:
  1. reset 时用打靶法 (scipy) 生成可行参考轨迹
  2. 观测: 当前状态 + 参考状态 + 追踪误差
  3. 动作: 1D 推力修正量 (沿参考方向)
  4. 奖励: 燃料最小化 + 终端成功

与 PD+RL 的区别: 参考来自打靶法而非 PD 在线制导律
"""
import gymnasium as gym
import numpy as np
from config import (
    R_ASTEROID, M0, M_DRY, M_FUEL0, T_MAX, I_SP, G0,
    DT, MAX_STEPS, SUCCESS_DIST, SUCCESS_VEL,
    ALTITUDE_MIN, ALTITUDE_MAX, ESCAPE_RADIUS,
)
from dynamics.integrator import dynamics, rk4_step
from dynamics.shooting import shooting_solve, get_ref_at_step

MAX_CORRECTION = T_MAX * 0.3


class ShootingRLEnv(gym.Env):
    """打靶法 + RL 燃料优化环境"""

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=-MAX_CORRECTION, high=MAX_CORRECTION, shape=(1,), dtype=np.float64
        )
        self.state = None
        self.ref_traj = None
        self.step_count = 0
        self.cumulative_fuel = 0.0
        self.target_pos = None
        self.shooting_fuel = 0.0

    def _surface_point(self):
        theta = self.np_random.uniform(0, 2 * np.pi)
        phi = self.np_random.uniform(0, np.pi / 2)
        return np.array([
            R_ASTEROID * np.cos(phi) * np.cos(theta),
            R_ASTEROID * np.cos(phi) * np.sin(theta),
            R_ASTEROID * np.sin(phi)
        ])

    def _random_initial_state(self):
        self.target_pos = self._surface_point()
        target_dir = self.target_pos / np.linalg.norm(self.target_pos)
        altitude = self.np_random.uniform(ALTITUDE_MIN, ALTITUDE_MAX)
        r = self.target_pos + target_dir * altitude
        approach_speed = self.np_random.uniform(0.5, 2.0)
        v = target_dir * (-approach_speed)
        v += self.np_random.standard_normal(3) * 0.2
        return np.array([r[0], r[1], r[2], v[0], v[1], v[2], M0])

    def _target_direction(self):
        delta = self.target_pos - self.state[0:3]
        dist = np.linalg.norm(delta)
        return delta / max(dist, 1e-6)

    def _get_obs(self):
        r = self.state[0:3]
        v = self.state[3:6]
        m = self.state[6]
        ref = get_ref_at_step(self.ref_traj, self.step_count)
        r_ref = ref[0:3]
        v_ref = ref[3:6]
        return np.array([
            r[0] / R_ASTEROID, r[1] / R_ASTEROID, r[2] / R_ASTEROID,
            v[0], v[1], v[2],
            (r[0] - r_ref[0]) / R_ASTEROID,
            (r[1] - r_ref[1]) / R_ASTEROID,
            (r[2] - r_ref[2]) / R_ASTEROID,
            v[0] - v_ref[0],
            v[1] - v_ref[1],
            v[2] - v_ref[2],
            m / M0,
        ], dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._random_initial_state()
        r0, v0 = self.state[0:3], self.state[3:6]
        self.ref_traj = shooting_solve(r0, v0, self.target_pos, M0)
        self.step_count = 0
        self.cumulative_fuel = 0.0
        return self._get_obs(), {}

    def step(self, action):
        correction = float(np.clip(action[0], -MAX_CORRECTION, MAX_CORRECTION))
        direction = self._target_direction()

        ref = get_ref_at_step(self.ref_traj, self.step_count)
        r_err_vec = self.state[0:3] - ref[0:3]
        v_err_vec = self.state[3:6] - ref[3:6]

        base_thrust_dir = -r_err_vec / max(np.linalg.norm(r_err_vec), 1e-6) if np.linalg.norm(r_err_vec) > 1e-6 else direction
        base_mag = np.clip(2.0 * np.linalg.norm(r_err_vec) + 1.0 * np.linalg.norm(v_err_vec), 0, T_MAX * 0.5)
        base_thrust = base_thrust_dir * base_mag + (-v_err_vec) * 0.5
        base_mag_total = np.linalg.norm(base_thrust)
        if base_mag_total > T_MAX * 0.6:
            base_thrust = base_thrust / base_mag_total * T_MAX * 0.6

        thrust = base_thrust + direction * correction
        thrust_mag = np.linalg.norm(thrust)
        if thrust_mag > T_MAX:
            thrust = thrust / thrust_mag * T_MAX
            thrust_mag = T_MAX

        self.state = rk4_step(dynamics, 0.0, self.state, thrust, DT)
        self.step_count += 1
        fuel_step = thrust_mag * DT / (I_SP * G0)
        self.cumulative_fuel += fuel_step

        r = self.state[0:3]
        v = self.state[3:6]
        m = self.state[6]
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)
        dist_to_target = np.linalg.norm(r - self.target_pos)

        r_err = np.linalg.norm(r - ref[0:3])
        v_err = np.linalg.norm(v - ref[3:6])

        reward = -0.3 * r_err / R_ASTEROID - 0.1 * v_err
        reward -= 2.0 * fuel_step / M_FUEL0
        reward -= 0.002

        terminated = False
        truncated = False

        if dist_to_target <= SUCCESS_DIST and v_norm <= SUCCESS_VEL:
            reward += 500.0
            reward -= 5.0 * self.cumulative_fuel / M_FUEL0
            terminated = True
        elif r_norm < R_ASTEROID:
            reward -= 500.0
            terminated = True
        elif r_norm > ESCAPE_RADIUS:
            reward -= 500.0
            terminated = True

        if self.step_count >= MAX_STEPS and not terminated:
            reward -= 200.0
            truncated = True

        info = {
            "dist_to_target": dist_to_target,
            "velocity": v_norm,
            "mass": m,
            "fuel_used": self.cumulative_fuel,
            "step_count": self.step_count,
            "thrust_mag": thrust_mag,
            "tracking_err_r": r_err,
            "tracking_err_v": v_err,
            "success": terminated and dist_to_target <= SUCCESS_DIST and v_norm <= SUCCESS_VEL,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass
