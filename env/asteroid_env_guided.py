"""
小行星附着 Gymnasium 环境 —— PD引导+RL修正 混合版本

智能体输出一个修正量 [-Δ, Δ]，叠加在 PD 制导基准上：
    thrust = pd_guidance(state) + rl_correction

PD 制导保证基本可行性，RL 学习燃料最优的精细修正。

观测空间（9维）：[Δr/R(3), v(3), m/m0, pd_thrust/max_thrust, dist/R]
动作空间（1维）：correction ∈ [-T_MAX*0.3, T_MAX*0.3]
"""
import gymnasium as gym
import numpy as np
from config import (
    R_ASTEROID, MU, M0, M_DRY, M_FUEL0, T_MAX, I_SP, G0,
    DT, MAX_STEPS, SUCCESS_DIST, SUCCESS_VEL,
    ALTITUDE_MIN, ALTITUDE_MAX, ESCAPE_RADIUS,
    K_DIST, K_VEL, K_PROGRESS, K_DECEL, LAMBDA_FUEL, GAMMA,
    R_SUCCESS, R_CRASH, R_ESCAPE, R_TIMEOUT, R_STEP,
)
from dynamics.integrator import dynamics, rk4_step

MAX_CORRECTION = T_MAX * 0.3


class AsteroidLandingEnvGuided(gym.Env):
    """PD 引导 + RL 修正的混合环境"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=-MAX_CORRECTION, high=MAX_CORRECTION, shape=(1,), dtype=np.float64
        )
        self.state = None
        self.prev_dist = None
        self.step_count = 0
        self.cumulative_fuel = 0.0
        self.target_pos = None

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

    def _pd_guidance(self):
        r = self.state[0:3]
        v = self.state[3:6]
        m = max(self.state[6], 1e-6)
        delta = self.target_pos - r
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return np.zeros(3)

        a_max = T_MAX / m
        desired_vel_mag = min(1.5, np.sqrt(2.0 * a_max * 0.3 * max(dist, 0.5)))

        target_dir = delta / dist
        vel_radial = np.dot(v, target_dir)
        vel_lateral = v - vel_radial * target_dir

        desired_vel_vector = target_dir * desired_vel_mag
        vel_error = desired_vel_vector - v

        desired_accel = 1.5 * vel_error + 0.8 * (-vel_lateral)
        accel_mag = np.linalg.norm(desired_accel)
        if accel_mag > a_max:
            desired_accel = desired_accel / accel_mag * a_max

        return desired_accel * m

    def _get_obs(self):
        r = self.state[0:3]
        v = self.state[3:6]
        m = self.state[6]
        delta_r = r - self.target_pos
        pd_vec = self._pd_guidance()
        pd_mag = np.linalg.norm(pd_vec)
        dist = np.linalg.norm(delta_r)
        return np.array([
            delta_r[0] / R_ASTEROID, delta_r[1] / R_ASTEROID, delta_r[2] / R_ASTEROID,
            v[0], v[1], v[2],
            m / M0,
            pd_mag / T_MAX,
            dist / R_ASTEROID,
        ], dtype=np.float64)

    def _target_direction(self):
        r = self.state[0:3]
        delta = self.target_pos - r
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return np.array([0.0, 0.0, 0.0])
        return delta / dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._random_initial_state()
        r = self.state[0:3]
        self.prev_dist = np.linalg.norm(r - self.target_pos)
        self.step_count = 0
        self.cumulative_fuel = 0.0
        return self._get_obs(), {}

    def step(self, action):
        correction = float(np.clip(action[0], -MAX_CORRECTION, MAX_CORRECTION))
        pd_thrust_vec = self._pd_guidance()
        direction = self._target_direction()

        thrust = pd_thrust_vec + direction * correction
        thrust_mag = np.linalg.norm(thrust)
        if thrust_mag > T_MAX:
            thrust = thrust / thrust_mag * T_MAX
            thrust_mag = T_MAX

        self.state = rk4_step(dynamics, 0.0, self.state, thrust, DT)
        self.step_count += 1
        self.cumulative_fuel += thrust_mag * DT / (I_SP * G0)

        r = self.state[0:3]
        v = self.state[3:6]
        m = self.state[6]
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)
        dist_to_target = np.linalg.norm(r - self.target_pos)

        reward = -K_DIST * dist_to_target / R_ASTEROID
        reward -= K_VEL * v_norm
        proximity = R_ASTEROID / (dist_to_target + 0.1)
        reward -= K_DECEL * v_norm * proximity * 0.1
        reward += K_PROGRESS * (self.prev_dist - dist_to_target) / R_ASTEROID
        reward -= LAMBDA_FUEL * thrust_mag * DT / (I_SP * G0 * M_FUEL0)
        reward += R_STEP

        self.prev_dist = dist_to_target

        terminated = False
        truncated = False

        if dist_to_target <= SUCCESS_DIST and v_norm <= SUCCESS_VEL:
            reward += R_SUCCESS
            terminated = True
        elif r_norm < R_ASTEROID:
            reward += R_CRASH
            terminated = True
        elif r_norm > ESCAPE_RADIUS:
            reward += R_ESCAPE
            terminated = True
        elif m <= M_DRY and thrust_mag < 1e-6:
            reward += R_CRASH
            terminated = True

        if self.step_count >= MAX_STEPS and not terminated:
            reward += R_TIMEOUT
            truncated = True

        info = {
            "dist_to_target": dist_to_target,
            "velocity": v_norm,
            "mass": m,
            "fuel_used": self.cumulative_fuel,
            "step_count": self.step_count,
            "thrust_mag": thrust_mag,
            "pd_thrust": float(np.linalg.norm(pd_thrust_vec)),
            "correction": correction,
            "success": terminated and dist_to_target <= SUCCESS_DIST and v_norm <= SUCCESS_VEL,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass
