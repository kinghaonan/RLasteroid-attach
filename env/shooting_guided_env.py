"""
按设计文档实现: 打靶法参考轨迹 + 势能塑形奖励 + PPO

架构:
  1. 打靶法在中心引力场求解参考轨迹 r_ref(t), v_ref(t), T_ref(t)
  2. 以参考推力为 feed-forward，RL 输出 1D 修正量 (已验证可行的模式)
  3. 势能函数 Φ(s) = -k1*||r-r_ref||/R - k2*||v-v_ref||
  4. 奖励 = γΦ(s')-Φ(s) + 燃料惩罚 + 终端

状态(14维): [Δr/R(3), Δv(3), r/R(3), v(3), m/m0, T_ref_mag/T_max]
动作(1维): 推力修正 ∈ [-T_max*0.3, T_max*0.3]
"""
import gymnasium as gym
import numpy as np
from config import (
    R_ASTEROID, M0, M_DRY, M_FUEL0, T_MAX, I_SP, G0,
    DT, MAX_STEPS, SUCCESS_DIST, SUCCESS_VEL,
    ALTITUDE_MIN, ALTITUDE_MAX, ESCAPE_RADIUS, GAMMA,
)
from dynamics.integrator import dynamics, rk4_step
from dynamics.shooting import shooting_solve, get_ref_at_step

MAX_CORRECTION = T_MAX * 0.3

# 从打靶法轨迹中提取参考推力 (简单估计: 根据速度变化反推)
def _estimate_ref_thrust(ref_traj, step):
    """从参考轨迹中估算参考推力"""
    if step + 1 < len(ref_traj):
        dv = ref_traj[step+1][3:6] - ref_traj[step][3:6]
        m = M0  # 近似常质量
        a_ref = dv / DT
        T_ref = m * a_ref
        return T_ref
    return np.zeros(3)


class ShootingGuidedEnv(gym.Env):
    """打靶法参考 + 势能塑形 + PPO (1D修正)"""

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=-MAX_CORRECTION, high=MAX_CORRECTION, shape=(1,), dtype=np.float64
        )
        self.state = None
        self.ref_traj = None
        self.step_count = 0
        self.cumulative_fuel = 0.0
        self.target_pos = None
        self.prev_potential = 0.0
        self._k1 = 2.0
        self._k2 = 1.0

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

    def _potential(self):
        r = self.state[0:3]
        v = self.state[3:6]
        ref = get_ref_at_step(self.ref_traj, self.step_count)
        r_err = np.linalg.norm(r - ref[0:3])
        v_err = np.linalg.norm(v - ref[3:6])
        return -self._k1 * r_err / R_ASTEROID - self._k2 * v_err

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
        T_ref = _estimate_ref_thrust(self.ref_traj, self.step_count)
        T_ref_mag = np.linalg.norm(T_ref)
        return np.array([
            (r[0] - r_ref[0]) / R_ASTEROID,
            (r[1] - r_ref[1]) / R_ASTEROID,
            (r[2] - r_ref[2]) / R_ASTEROID,
            v[0] - v_ref[0],
            v[1] - v_ref[1],
            v[2] - v_ref[2],
            r[0] / R_ASTEROID, r[1] / R_ASTEROID, r[2] / R_ASTEROID,
            v[0], v[1], v[2],
            m / M0,
            T_ref_mag / T_MAX,
        ], dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._random_initial_state()
        r0, v0 = self.state[0:3], self.state[3:6]
        self.ref_traj = shooting_solve(r0, v0, self.target_pos, M0)
        self.step_count = 0
        self.cumulative_fuel = 0.0
        self.prev_potential = self._potential()
        return self._get_obs(), {}

    def step(self, action):
        correction = float(np.clip(action[0], -MAX_CORRECTION, MAX_CORRECTION))
        direction = self._target_direction()

        T_ref = _estimate_ref_thrust(self.ref_traj, self.step_count)
        ref_mag = np.linalg.norm(T_ref)
        if ref_mag > T_MAX * 0.8:
            T_ref = T_ref / ref_mag * T_MAX * 0.8
            ref_mag = T_MAX * 0.8

        thrust = T_ref + direction * correction
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

        current_potential = self._potential()
        reward = GAMMA * current_potential - self.prev_potential
        reward -= 0.12 * fuel_step / M_FUEL0
        self.prev_potential = current_potential

        terminated = False
        truncated = False

        if dist_to_target <= SUCCESS_DIST and v_norm <= SUCCESS_VEL:
            reward += 1000.0
            terminated = True
        elif r_norm < R_ASTEROID:
            reward -= 1000.0
            terminated = True
        elif r_norm > ESCAPE_RADIUS:
            reward -= 1000.0
            terminated = True

        if self.step_count >= MAX_STEPS and not terminated:
            reward -= 500.0
            truncated = True

        info = {
            "dist_to_target": dist_to_target,
            "velocity": v_norm,
            "mass": m,
            "fuel_used": self.cumulative_fuel,
            "step_count": self.step_count,
            "thrust_mag": thrust_mag,
            "success": terminated and dist_to_target <= SUCCESS_DIST and v_norm <= SUCCESS_VEL,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

