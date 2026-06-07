"""
评估脚本 —— 加载训练好的 PPO 模型，运行评估回合，生成可视化

用法:
    python eval.py                          # 使用最新模型
    python eval.py --model ./logs/best_model/best_model.zip  # 指定模型
    python eval.py --n 50                   # 运行 50 个评估回合
"""
import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from env.asteroid_env_guided import AsteroidLandingEnvGuided
from utils.visualization import (
    plot_trajectory_3d,
    plot_state_history,
    plot_action_history,
    plot_training_curves,
)
from config import T_MAX, R_ASTEROID, DT, SUCCESS_DIST, SUCCESS_VEL


def run_evaluation(model_path, vec_normalize_path, n_episodes=20, render_trajectory=True):
    """运行评估并收集统计"""
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行 train.py 训练模型")
        return

    env = AsteroidLandingEnvGuided()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    if os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"警告: VecNormalize 文件不存在 ({vec_normalize_path})")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)

    model = PPO.load(model_path, env=env)

    success_count = 0
    crash_count = 0
    escape_count = 0
    timeout_count = 0
    fuel_used_list = []
    steps_list = []
    final_dist_list = []
    final_vel_list = []

    best_trajectory = None
    best_reward = -np.inf
    best_actions = None

    print(f"运行 {n_episodes} 个评估回合...")

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        trajectory = []
        actions = []

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            ep_reward += reward[0]
            done_flag = dones[0]

            env_instance = env.envs[0].env
            trajectory.append(env_instance.state.copy())
            actions.append(action[0].copy())

            if done_flag:
                break

        info_dict = info[0]
        success = info_dict.get("success", False)
        dist = info_dict.get("dist_to_target", 999)
        vel = info_dict.get("velocity", 999)
        fuel = info_dict.get("fuel_used", 0)
        steps = info_dict.get("step_count", 0)

        if success:
            success_count += 1
        elif dist > 20 * R_ASTEROID:
            escape_count += 1
        elif dist < R_ASTEROID or (dist <= R_ASTEROID + 1 and vel > 1.0):
            crash_count += 1
        else:
            timeout_count += 1

        fuel_used_list.append(fuel)
        steps_list.append(steps)
        final_dist_list.append(dist)
        final_vel_list.append(vel)

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_trajectory = trajectory
            best_actions = actions

        print(f"  Episode {ep+1:3d}: "
              f"{'成功' if success else '失败':4s}  "
              f"奖励={ep_reward:8.2f}  "
              f"步数={steps:4d}  "
              f"燃料={fuel:6.2f}kg  "
              f"终距={dist:8.1f}m  "
              f"终速={vel:6.3f}m/s")

    print(f"\n{'='*60}")
    print(f"评估完成 ({n_episodes} 回合)")
    print(f"  成功率: {success_count}/{n_episodes} = {success_count/n_episodes*100:.1f}%")
    print(f"  撞击: {crash_count},  逃逸: {escape_count},  超时: {timeout_count}")
    print(f"  平均燃料消耗: {np.mean(fuel_used_list):.2f} ± {np.std(fuel_used_list):.2f} kg")
    print(f"  平均步数: {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
    print(f"  平均终距: {np.mean(final_dist_list):.1f} ± {np.std(final_dist_list):.1f} m")
    print(f"  平均终速: {np.mean(final_vel_list):.4f} ± {np.std(final_vel_list):.4f} m/s")
    print(f"{'='*60}")

    if render_trajectory and best_trajectory is not None:
        env_instance = env.envs[0].env
        target_pos = env_instance.target_pos
        print("\n绘制最佳轨迹...")
        plot_trajectory_3d(best_trajectory, target_pos, R_ASTEROID,
                          save_path="./logs/best_trajectory_3d.png")
        plot_state_history(best_trajectory, DT,
                          save_path="./logs/state_history.png")
        plot_action_history(best_actions, DT, T_MAX,
                           save_path="./logs/action_history.png")

    plot_training_curves(save_path="./logs/training_curves.png")

    return {
        "success_rate": success_count / n_episodes,
        "avg_fuel": np.mean(fuel_used_list),
        "avg_steps": np.mean(steps_list),
        "avg_final_dist": np.mean(final_dist_list),
        "avg_final_vel": np.mean(final_vel_list),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="小行星附着 RL 策略评估")
    parser.add_argument("--model", type=str, default="./logs/asteroid_landing_ppo_final.zip",
                        help="模型文件路径")
    parser.add_argument("--vecnorm", type=str, default="./logs/vec_normalize.pkl",
                        help="VecNormalize 文件路径")
    parser.add_argument("--n", type=int, default=50,
                        help="评估回合数")
    parser.add_argument("--no-plot", action="store_true",
                        help="不生成可视化图表")
    args = parser.parse_args()

    run_evaluation(args.model, args.vecnorm, args.n, render_trajectory=not args.no_plot)
