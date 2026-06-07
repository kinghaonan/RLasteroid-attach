"""最终评估脚本"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env.asteroid_env_guided import AsteroidLandingEnvGuided

env = AsteroidLandingEnvGuided()
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecNormalize.load("./logs/vec_normalize.pkl", env)
env.training = False
env.norm_reward = False
model = PPO.load("./logs/asteroid_landing_ppo_final.zip", env=env)

succ = 0
fuels = []
steps_list = []
dists = []
vels = []
n = 30

for ep in range(n):
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)
        if dones[0]:
            d = info[0]
            fuels.append(d["fuel_used"])
            steps_list.append(d["step_count"])
            dists.append(d["dist_to_target"])
            vels.append(d["velocity"])
            if d["success"]:
                succ += 1
            print(f"  Ep {ep+1:2d}: "
                  f"success={d['success']}, steps={d['step_count']:3d}, "
                  f"fuel={d['fuel_used']:.2f}kg, "
                  f"dist={d['dist_to_target']:.1f}m, "
                  f"vel={d['velocity']:.3f}m/s")
            break

print(f"\n{'='*50}")
print(f"成功率: {succ}/{n} = {succ/n*100:.0f}%")
print(f"平均燃料: {np.mean(fuels):.2f} +/- {np.std(fuels):.2f} kg")
print(f"平均步数: {np.mean(steps_list):.1f} +/- {np.std(steps_list):.1f}")
print(f"平均终距: {np.mean(dists):.1f} +/- {np.std(dists):.1f} m")
print(f"平均终速: {np.mean(vels):.4f} +/- {np.std(vels):.4f} m/s")
print(f"最大推力: 50N, 初始质量: 500kg, 干质量: 300kg")
