# 小行星附着最优轨迹 RL 实施计划

> **For Claude:** 使用 subagent-driven-development 模式逐任务实现。

**目标：** 构建基于 PPO 强化学习的小行星附着轨迹优化系统，实现燃料最优的自主软着陆。

**架构：** Gymnasium 环境封装中心引力场动力学 + RK4 积分，stable-baselines3 PPO 训练，势能奖励塑形引导学习。

**技术栈：** Python 3.9+, Gymnasium, stable-baselines3, PyTorch, NumPy, Matplotlib

---

## 物理参数设计

### 小行星（中心引力场，R=500m 岩石行星，无自转）
- 半径 R = 500 m
- 密度 ρ = 2500 kg/m³
- 万有引力常数 G = 6.67430×10⁻¹¹ m³/(kg·s²)
- 质量 M = (4/3)πR³ρ = 1.309×10¹² kg
- 引力参数 μ = GM = 87.37 m³/s²
- 表面重力加速度 g_surf = μ/R² = 3.495×10⁻⁴ m/s²（极弱引力场）

### 航天器
- 初始质量 m₀ = 500 kg
- 干质量 m_p = 300 kg（含 200 kg 燃料）
- 最大推力 T_max = 50 N
- 比冲 I_sp = 300 s
- g₀ = 9.80665 m/s²
- 燃料消耗率（满推力）ṁ = T_max/(I_sp·g₀) ≈ 0.017 kg/s

### 动力学方程（中心引力 + 无自转）
```
dr/dt = v
dv/dt = -μ·r/|r|³ + T/m
dm/dt = -|T|/(I_sp·g₀)
```
注意：ω=0，无科氏力和离心力。

---

## 项目结构

```
code/
├── config.py                 # 全局物理参数和超参数
├── dynamics/
│   ├── __init__.py
│   ├── gravity.py            # 中心引力场加速度计算
│   └── integrator.py         # RK4 数值积分
├── env/
│   ├── __init__.py
│   ├── asteroid_env.py       # Gymnasium 环境（核心）
├── train.py                  # PPO 训练脚本
├── eval.py                   # 评估与可视化
├── utils/
│   ├── __init__.py
│   └── visualization.py      # 轨迹绘制、训练曲线
└── requirements.txt
```

---

## Task 1: 项目骨架与配置

**文件：**
- 创建: `config.py`
- 创建: `requirements.txt`

**config.py** 包含所有物理参数和 RL 超参数：

```python
# === 小行星参数 ===
R_ASTEROID = 500.0          # 半径 [m]
RHO_ASTEROID = 2500.0       # 密度 [kg/m³]
G = 6.67430e-11             # 万有引力常数
# 导出量
import numpy as np
M_ASTEROID = RHO_ASTEROID * (4/3) * np.pi * R_ASTEROID**3  # ≈1.309e12 kg
MU = G * M_ASTEROID         # 引力参数 ≈87.37 m³/s²

# === 航天器参数 ===
M0 = 500.0                  # 初始质量 [kg]
M_DRY = 300.0               # 干质量 [kg]
T_MAX = 50.0                # 最大推力 [N]
I_SP = 300.0                # 比冲 [s]
G0 = 9.80665                # 地球重力加速度

# === 环境参数 ===
DT = 1.0                    # 积分步长 [s]
MAX_STEPS = 2000            # 最大步数
SUCCESS_DIST = 10.0         # 成功着陆距离 [m]
SUCCESS_VEL = 0.1           # 成功着陆速度 [m/s]
ALTITUDE_RANGE = (800, 2500) # 初始高度范围 [m]（距表面）
ESCAPE_RADIUS = 10 * R_ASTEROID  # 逃逸半径

# === 奖励参数 ===
K_POS = 0.01               # 位置势能系数
K_VEL = 1.0                # 速度势能系数
LAMBDA_FUEL = 0.1          # 燃料惩罚系数
GAMMA = 0.99               # 折扣因子
R_SUCCESS = 500.0          # 成功奖励
R_CRASH = -500.0           # 撞击惩罚
R_ESCAPE = -500.0          # 逃逸惩罚
R_TIMEOUT = -200.0         # 超时惩罚

# === PPO 超参数 ===
N_ENVS = 8                 # 并行环境数
N_STEPS = 2048             # 每轮步数
BATCH_SIZE = 256
N_EPOCHS = 10
LEARNING_RATE = 3e-4
TOTAL_TIMESTEPS = 2_000_000
```

**requirements.txt**:
```
gymnasium>=0.29.0
stable-baselines3>=2.3.0
numpy>=1.24.0
matplotlib>=3.7.0
torch>=2.0.0
```

---

## Task 2: 动力学模块

**文件：**
- 创建: `dynamics/__init__.py`
- 创建: `dynamics/gravity.py`
- 创建: `dynamics/integrator.py`

### 2.1 中心引力场 (`gravity.py`)

```python
import numpy as np
from config import MU

def gravity_accel(r: np.ndarray) -> np.ndarray:
    """中心引力场加速度 a = -μ·r/|r|³"""
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        return np.zeros(3)
    return -MU * r / r_norm**3
```

### 2.2 RK4 积分器 (`integrator.py`)

```python
import numpy as np
from dynamics.gravity import gravity_accel
from config import I_SP, G0

def thrust_accel(thrust: np.ndarray, mass: float) -> np.ndarray:
    """推力加速度，限制幅值"""
    return thrust / mass

def mass_flow_rate(thrust_mag: float) -> float:
    """质量变化率"""
    return -thrust_mag / (I_SP * G0)

def dynamics(t: float, state: np.ndarray, thrust: np.ndarray) -> np.ndarray:
    """
    全耦合动力学右端函数
    state = [rx, ry, rz, vx, vy, vz, m]
    thrust = [Tx, Ty, Tz]  (N)
    返回: dstate/dt
    """
    r = state[0:3]
    v = state[3:6]
    m = state[6]
    
    # 限制推力大小
    thrust_mag = np.linalg.norm(thrust)
    
    # 引力加速度
    a_grav = gravity_accel(r)
    
    # 推力加速度
    a_thrust = thrust / m
    
    # 质量变化
    dm = mass_flow_rate(thrust_mag)
    
    return np.array([v[0], v[1], v[2],
                     a_grav[0] + a_thrust[0],
                     a_grav[1] + a_thrust[1],
                     a_grav[2] + a_thrust[2],
                     dm])

def rk4_step(f, t: float, state: np.ndarray, thrust: np.ndarray, dt: float) -> np.ndarray:
    """四阶龙格-库塔积分一步"""
    k1 = f(t, state, thrust)
    k2 = f(t + dt/2, state + dt/2 * k1, thrust)
    k3 = f(t + dt/2, state + dt/2 * k2, thrust)
    k4 = f(t + dt, state + dt * k3, thrust)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
```

---

## Task 3: Gymnasium 环境

**文件：**
- 创建: `env/__init__.py`
- 创建: `env/asteroid_env.py`

### 3.1 环境设计

**状态空间（观测）** `obs ∈ ℝ⁷`:
```
[rx/R, ry/R, rz/R, vx, vy, vz, m/m₀]
```
- 位置用 R 归一化（无量纲）
- 速度保持原单位（m/s，范围小）
- 质量用 m₀ 归一化

**动作空间** `act ∈ ℝ³`:
```
[Tx, Ty, Tz]  每维 ∈ [-T_max, T_max]
```

**奖励函数（势能塑形 + 燃料惩罚 + 终端奖惩）**:

势能函数: `Φ(s) = -k_pos·|r - r_target|/R - k_vel·|v|`

单步奖励:
```
R_step = γ·Φ(s') - Φ(s)  (势能差)
         - λ_fuel·|T|·dt/(I_sp·g₀·m_fuel₀)  (燃料惩罚)
```

终端奖励:
- 成功（|r-r_target|<10m & |v|<0.1m/s）: +500
- 撞击（|r|<R）: -500
- 逃逸（|r|>10R）: -500
- 超时: -200

**终止条件**:
1. 成功着陆
2. 撞击表面（|r| < R 或 z 过低判断在体坐标系下）
3. 逃逸（|r| > 10R）
4. 燃料耗尽（m ≤ m_dry，且无推力能力）
5. 超时（步数 > MAX_STEPS）

**初始状态随机化**:
- 初始位置：在目标点上方随机高度（800-2500m），有随机水平偏移
- 初始速度：小的随机速度 [-0.2, 0.2] m/s
- 初始质量：m₀

**环境重置**:
- 每次 reset 随机化初始条件（在合理范围内）

**核心实现骨架**:

```python
import gymnasium as gym
import numpy as np
from config import *
from dynamics.integrator import dynamics, rk4_step

class AsteroidLandingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 观测空间：[rx/R, ry/R, rz/R, vx, vy, vz, m/m0]  7维
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf]*7),
            high=np.array([np.inf]*7),
            dtype=np.float32
        )
        # 动作空间：[Tx, Ty, Tz]  连续推力分量
        self.action_space = gym.spaces.Box(
            low=-T_MAX, high=T_MAX, shape=(3,), dtype=np.float32
        )
        self.state = None  # [rx, ry, rz, vx, vy, vz, m]
        self.step_count = 0
        self.target_pos = None

    def _get_obs(self):
        r, v, m = self.state[:3], self.state[3:6], self.state[6]
        return np.array([
            r[0]/R_ASTEROID, r[1]/R_ASTEROID, r[2]/R_ASTEROID,
            v[0], v[1], v[2],
            m / M0
        ], dtype=np.float32)

    def _compute_reward(self, thrust):
        r, v = self.state[:3], self.state[3:6]
        # 势能函数
        phi = lambda r,v: -K_POS*np.linalg.norm(r-self.target_pos)/R_ASTEROID - K_VEL*np.linalg.norm(v)
        # 这里需要在 step 中计算当前势能和上一步势能的差值
        ...

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 随机初始化...
        ...
        return self._get_obs(), {}

    def step(self, action):
        thrust = np.clip(action, -T_MAX, T_MAX)
        # RK4 积分
        self.state = rk4_step(dynamics, 0, self.state, thrust, DT)
        self.step_count += 1
        # 计算奖励、终止条件
        ...
        return obs, reward, terminated, truncated, info
```

详细实现见 Task 3 具体代码。

---

## Task 4: PPO 训练脚本

**文件：**
- 创建: `train.py`

使用 stable-baselines3 的 PPO 实现：

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from env.asteroid_env import AsteroidLandingEnv
from config import *

def make_env():
    return AsteroidLandingEnv()

def train():
    # 向量化环境
    env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # 评估环境
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, 
                            training=False, clip_obs=10.0)
    
    # 回调
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model/',
                                 log_path='./logs/', eval_freq=10000,
                                 deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./logs/checkpoints/')
    
    # PPO 模型
    model = PPO("MlpPolicy", env, verbose=1,
                n_steps=N_STEPS, batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE,
                gamma=GAMMA, tensorboard_log="./logs/tensorboard/")
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_callback, checkpoint_callback])
    
    # 保存
    model.save("asteroid_landing_ppo")
    env.save("vec_normalize.pkl")

if __name__ == "__main__":
    train()
```

---

## Task 5: 评估与可视化

**文件：**
- 创建: `eval.py`
- 创建: `utils/__init__.py`
- 创建: `utils/visualization.py`

评估脚本加载训练好的模型，运行多条 episode，绘制：
1. 3D 轨迹图（含小行星球体）
2. 位置/速度/质量随时间变化曲线
3. 推力指令曲线
4. 训练 reward 曲线（从 TensorBoard 日志读取）

---

## 执行顺序

| 任务 | 内容 | 依赖 |
|------|------|------|
| Task 1 | 配置文件和项目骨架 | 无 |
| Task 2 | 动力学模块 | Task 1 |
| Task 3 | Gymnasium 环境 | Task 2 |
| Task 4 | PPO 训练脚本 | Task 3 |
| Task 5 | 评估与可视化 | Task 4 |

---

## 注意事项

1. **极弱引力场**：小行星表面重力仅 ~3.5×10⁻⁴ m/s²，推力足以轻易克服引力。RL 的核心挑战是学会**省燃料 + 精准着陆**，而非克服引力。
2. **初始条件随机化**：训练时随机初始位置和速度，增强策略泛化能力。
3. **势能塑形**：使用基于势能的奖励塑形加速收敛，数学上不改变最优策略。
4. **VecNormalize**：使用环境归一化稳定训练。
5. **训练监控**：使用 TensorBoard 监控 reward、episode length、成功率。
