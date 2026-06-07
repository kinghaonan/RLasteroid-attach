"""着陆轨迹与训练结果可视化"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env.asteroid_env_guided import AsteroidLandingEnvGuided
from config import R_ASTEROID, T_MAX, M0, M_DRY, DT, SUCCESS_DIST, SUCCESS_VEL, GAMMA

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs("./logs/figures", exist_ok=True)


def load_model():
    env = AsteroidLandingEnvGuided()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load("./logs/vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False
    model = PPO.load("./logs/asteroid_landing_ppo_final.zip", env=env)
    return model, env


def collect_trajectory(model, env):
    obs = env.reset()
    trajectory = []
    actions_raw = []
    rewards = []
    infos = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)

        inst = env.envs[0].env
        trajectory.append(inst.state.copy())
        actions_raw.append(action[0].copy())
        rewards.append(reward[0])
        infos.append(info[0])

        if dones[0]:
            break

    return trajectory, actions_raw, rewards, infos


def plot_trajectory_3d(trajectory, target_pos, save_path):
    """3D 着陆轨迹图"""
    traj = np.array([s[0:3] for s in trajectory])
    fig = plt.figure(figsize=(12, 10), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = R_ASTEROID * np.outer(np.cos(u), np.sin(v))
    y = R_ASTEROID * np.outer(np.sin(u), np.sin(v))
    z = R_ASTEROID * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="tan", alpha=0.25, linewidth=0, antialiased=True)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.08, linewidth=0.3, rstride=4, cstride=4)

    colors = plt.cm.viridis(np.linspace(0, 1, len(traj)))
    for i in range(len(traj) - 1):
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], traj[i:i+2, 2],
                color=colors[i], linewidth=2.0)

    ax.scatter(*traj[0], color="lime", s=120, marker="o",
               edgecolors="darkgreen", linewidth=1.5, label="Initial Position", zorder=10)
    ax.scatter(*traj[-1], color="red", s=150, marker="*",
               edgecolors="darkred", linewidth=1.5, label="Landing Point", zorder=10)
    ax.scatter(*target_pos, color="gold", s=200, marker="X",
               edgecolors="darkorange", linewidth=1.5, label="Target Site", zorder=10)

    r_max = max(np.max(np.abs(traj)), R_ASTEROID * 1.6)
    ax.set_xlim([-r_max, r_max])
    ax.set_ylim([-r_max, r_max])
    ax.set_zlim([-r_max, r_max])
    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)
    ax.set_zlabel("Z [m]", fontsize=12)
    ax.set_title("Asteroid Landing 3D Trajectory\n(Central Gravity, R=500m, PPO + PD Guidance)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  3D 轨迹图已保存: {save_path}")


def plot_state_history(trajectory, save_path):
    """位置、速度、质量随时间变化"""
    traj = np.array(trajectory)
    t = np.arange(len(traj)) * DT
    r = traj[:, 0:3]
    v = traj[:, 3:6]
    m = traj[:, 6]
    r_mag = np.linalg.norm(r, axis=1)
    v_mag = np.linalg.norm(v, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    axes[0, 0].plot(t, r[:, 0], "#2196F3", linewidth=1.5, label="X")
    axes[0, 0].plot(t, r[:, 1], "#4CAF50", linewidth=1.5, label="Y")
    axes[0, 0].plot(t, r[:, 2], "#FF9800", linewidth=1.5, label="Z")
    axes[0, 0].set_xlabel("Time [s]", fontsize=11)
    axes[0, 0].set_ylabel("Position [m]", fontsize=11)
    axes[0, 0].set_title("Position Components", fontsize=12, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, r_mag, "#1565C0", linewidth=2.0, label="|r|")
    axes[0, 1].axhline(y=R_ASTEROID, color="red", linestyle="--", alpha=0.6, linewidth=1.2,
                       label=f"Surface (R={R_ASTEROID}m)")
    axes[0, 1].axhline(y=R_ASTEROID + SUCCESS_DIST, color="green", linestyle=":", alpha=0.6, linewidth=1.0,
                       label=f"Success zone (R+{SUCCESS_DIST}m)")
    axes[0, 1].fill_between(t, R_ASTEROID, R_ASTEROID + SUCCESS_DIST,
                            alpha=0.1, color="green")
    axes[0, 1].set_xlabel("Time [s]", fontsize=11)
    axes[0, 1].set_ylabel("Distance from Center [m]", fontsize=11)
    axes[0, 1].set_title("Radial Distance", fontsize=12, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t, v[:, 0], "#2196F3", linewidth=1.2, alpha=0.7, label="Vx")
    axes[1, 0].plot(t, v[:, 1], "#4CAF50", linewidth=1.2, alpha=0.7, label="Vy")
    axes[1, 0].plot(t, v[:, 2], "#FF9800", linewidth=1.2, alpha=0.7, label="Vz")
    axes[1, 0].plot(t, v_mag, "k-", linewidth=2.0, label="|V|")
    axes[1, 0].axhline(y=SUCCESS_VEL, color="green", linestyle=":", alpha=0.5, linewidth=1.0,
                       label=f"Success limit ({SUCCESS_VEL}m/s)")
    axes[1, 0].set_xlabel("Time [s]", fontsize=11)
    axes[1, 0].set_ylabel("Velocity [m/s]", fontsize=11)
    axes[1, 0].set_title("Velocity Components", fontsize=12, fontweight="bold")
    axes[1, 0].legend(fontsize=9, ncol=2)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t, m, "#E91E63", linewidth=2.0)
    axes[1, 1].axhline(y=M_DRY, color="gray", linestyle="--", alpha=0.5, linewidth=1.0,
                       label=f"Dry mass ({M_DRY}kg)")
    axes[1, 1].fill_between(t, M_DRY, M0, alpha=0.1, color="red")
    axes[1, 1].set_xlabel("Time [s]", fontsize=11)
    axes[1, 1].set_ylabel("Mass [kg]", fontsize=11)
    axes[1, 1].set_title("Mass Variation", fontsize=12, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([M_DRY - 5, M0 + 5])

    fig.suptitle("Asteroid Landing State History", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  状态历史图已保存: {save_path}")


def plot_thrust_history(actions, infos, save_path):
    acts = np.array(actions)
    t = np.arange(len(acts)) * DT
    thrust_mags = [abs(float(a[0])) for a in acts]

    n_infos = len(infos)
    n_acts = len(acts)
    if n_infos < n_acts:
        infos_ext = infos + [infos[-1]] * (n_acts - n_infos)
    else:
        infos_ext = infos[:n_acts]

    pd_thrusts = [info.get("pd_thrust", 0) for info in infos_ext]
    corrections = [info.get("correction", 0) for info in infos_ext]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=150)

    axes[0].fill_between(t, 0, thrust_mags, alpha=0.3, color="#2196F3", label="Actual Thrust")
    axes[0].plot(t, pd_thrusts, "#FF5722", linewidth=1.5, linestyle="--", label="PD Baseline")
    axes[0].axhline(y=T_MAX, color="red", linestyle=":", alpha=0.4, linewidth=1.0, label=f"T_max ({T_MAX}N)")
    axes[0].set_xlabel("Time [s]", fontsize=11)
    axes[0].set_ylabel("Thrust Magnitude [N]", fontsize=11)
    axes[0].set_title("Thrust Command vs PD Baseline", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(t, 0, corrections, alpha=0.4,
                         color=["#4CAF50" if c >= 0 else "#F44336" for c in corrections])
    axes[1].plot(t, corrections, "#333333", linewidth=1.0)
    axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.8)
    axes[1].set_xlabel("Time [s]", fontsize=11)
    axes[1].set_ylabel("RL Correction [N]", fontsize=11)
    axes[1].set_title("RL Correction Signal (on top of PD)", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Thrust Command Analysis", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  推力曲线图已保存: {save_path}")


def plot_training_curves(save_path):
    """训练奖励曲线"""
    eval_file = "./logs/eval/evaluations.npz"
    if not os.path.exists(eval_file):
        print(f"  未找到评估日志: {eval_file}, 跳过训练曲线")
        return

    data = np.load(eval_file)
    timesteps = data["timesteps"]
    results = data["results"]
    ep_lengths = data["ep_lengths"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    mean_r = results.mean(axis=1)
    std_r = results.std(axis=1)
    axes[0].plot(timesteps / 1e6, mean_r, "#2196F3", linewidth=2.0)
    axes[0].fill_between(timesteps / 1e6, mean_r - std_r, mean_r + std_r,
                         alpha=0.2, color="#2196F3")
    axes[0].set_xlabel("Training Steps [M]", fontsize=11)
    axes[0].set_ylabel("Mean Episode Reward", fontsize=11)
    axes[0].set_title("Evaluation Reward Curve", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.3, linewidth=0.8)

    mean_l = ep_lengths.mean(axis=1)
    std_l = ep_lengths.std(axis=1)
    axes[1].plot(timesteps / 1e6, mean_l, "#4CAF50", linewidth=2.0)
    axes[1].fill_between(timesteps / 1e6, mean_l - std_l, mean_l + std_l,
                         alpha=0.2, color="#4CAF50")
    axes[1].set_xlabel("Training Steps [M]", fontsize=11)
    axes[1].set_ylabel("Mean Episode Length [steps]", fontsize=11)
    axes[1].set_title("Episode Length", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    success_rate = np.array([np.sum(r > 4000) / len(r) * 100 for r in results])
    axes[2].plot(timesteps / 1e6, success_rate, "#FF9800", linewidth=2.0)
    axes[2].set_xlabel("Training Steps [M]", fontsize=11)
    axes[2].set_ylabel("Success Rate [%]", fontsize=11)
    axes[2].set_title("Estimated Landing Success Rate", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 105])

    fig.suptitle("PPO Training Progress (PD-Guided)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  训练曲线图已保存: {save_path}")


def plot_summary_panel(stats, save_path):
    """综合评估面板"""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.axis("off")

    text = (
        "Asteroid Landing RL - Final Evaluation Summary\n"
        "=" * 55 + "\n\n"
        f"  Success Rate:         {stats['success_rate']:.0f}%  ({stats['n_success']}/{stats['n_total']})\n"
        f"  Avg Fuel Consumption:  {stats['avg_fuel']:.2f} \u00b1 {stats['std_fuel']:.2f} kg\n"
        f"  Avg Landing Time:      {stats['avg_steps']:.1f} \u00b1 {stats['std_steps']:.1f} s\n"
        f"  Avg Final Distance:    {stats['avg_dist']:.1f} \u00b1 {stats['std_dist']:.1f} m\n"
        f"  Avg Final Velocity:    {stats['avg_vel']:.4f} \u00b1 {stats['std_vel']:.4f} m/s\n\n"
        "Configuration\n"
        "-" * 55 + "\n"
        f"  Asteroid:  R=500m, rocky (\u03c1=2500 kg/m\u00b3), central gravity\n"
        f"  Spacecraft: m0=500kg, T_max=50N, Isp=300s\n"
        f"  Algorithm:  PPO + PD Guidance (PD base + RL correction)\n"
        f"  Training:   2M steps, 8 parallel envs, entropy_coef=0.05"
    )

    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=11,
            verticalalignment="center", horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#F5F5F5",
                      edgecolor="#333333", alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  综合面板已保存: {save_path}")


def main():
    print("=" * 55)
    print("  小行星附着 RL 轨迹可视化")
    print("=" * 55)

    print("\n[1/5] 加载模型...")
    model, env = load_model()

    print("\n[2/5] 收集最优轨迹...")
    best_traj, best_acts, best_rews, best_infos = None, None, None, None
    best_reward = -np.inf
    for _ in range(10):
        traj, acts, rews, infos = collect_trajectory(model, env)
        total_r = sum(rews)
        if total_r > best_reward:
            best_reward = total_r
            best_traj = traj
            best_acts = acts
            best_rews = rews
            best_infos = infos
    print(f"  最优轨迹: {len(best_traj)} 步, 累计奖励: {best_reward:.1f}")

    target = env.envs[0].env.target_pos

    print("\n[3/5] 生成可视化图表...")
    plot_trajectory_3d(best_traj, target, "./logs/figures/trajectory_3d.png")
    plot_state_history(best_traj, "./logs/figures/state_history.png")
    plot_thrust_history(best_acts, best_infos, "./logs/figures/thrust_history.png")
    plot_training_curves("./logs/figures/training_curves.png")

    print("\n[4/5] 运行批量评估 (50 episodes)...")
    succ = 0
    fuels, steps_l, dists_l, vels_l = [], [], [], []
    for ep in range(50):
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            if dones[0]:
                d = info[0]
                fuels.append(d["fuel_used"])
                steps_l.append(d["step_count"])
                dists_l.append(d["dist_to_target"])
                vels_l.append(d["velocity"])
                if d["success"]:
                    succ += 1
                break

    stats = {
        "n_total": 50, "n_success": succ,
        "success_rate": succ / 50 * 100,
        "avg_fuel": np.mean(fuels), "std_fuel": np.std(fuels),
        "avg_steps": np.mean(steps_l), "std_steps": np.std(steps_l),
        "avg_dist": np.mean(dists_l), "std_dist": np.std(dists_l),
        "avg_vel": np.mean(vels_l), "std_vel": np.std(vels_l),
    }

    print(f"  成功率: {stats['success_rate']:.0f}%")
    print(f"  平均燃料: {stats['avg_fuel']:.2f} kg")
    print(f"  平均着陆时间: {stats['avg_steps']:.1f} s")

    print("\n[5/5] 生成综合评估面板...")
    plot_summary_panel(stats, "./logs/figures/summary_panel.png")

    print("\n" + "=" * 55)
    print("  可视化完成! 图表保存在 ./logs/figures/")
    print("=" * 55)


if __name__ == "__main__":
    main()
