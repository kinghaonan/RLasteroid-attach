"""可视化工具 —— 3D 轨迹、状态曲线、训练曲线"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_trajectory_3d(trajectory, target_pos, asteroid_radius=500.0, save_path=None):
    """
    绘制 3D 着陆轨迹

    参数
    ----
    trajectory : list of np.ndarray
        每个元素为 state [rx, ry, rz, vx, vy, vz, m]
    target_pos : np.ndarray
        着陆目标点 (3,)
    asteroid_radius : float
        小行星半径
    save_path : str or None
        保存路径
    """
    traj = np.array([s[0:3] for s in trajectory])
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x = asteroid_radius * np.outer(np.cos(u), np.sin(v))
    y = asteroid_radius * np.outer(np.sin(u), np.sin(v))
    z = asteroid_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="gray", alpha=0.3, linewidth=0)

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "b-", linewidth=1.5, label="着陆轨迹")
    ax.scatter(*traj[0], color="green", s=80, marker="o", label="初始位置", zorder=5)
    ax.scatter(*traj[-1], color="red", s=80, marker="*", label="终端位置", zorder=5)
    ax.scatter(*target_pos, color="orange", s=100, marker="x", label="目标着陆点", zorder=5)

    max_val = max(np.max(np.abs(traj)), asteroid_radius * 1.5)
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("小行星附着 3D 轨迹")
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_state_history(trajectory, dt=1.0, save_path=None):
    """
    绘制位置、速度、质量随时间变化

    参数
    ----
    trajectory : list of np.ndarray
    dt : float
        时间步长
    save_path : str or None
    """
    traj = np.array(trajectory)
    t = np.arange(len(traj)) * dt
    r_mag = np.linalg.norm(traj[:, 0:3], axis=1)
    v_mag = np.linalg.norm(traj[:, 3:6], axis=1)
    mass = traj[:, 6]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(t, traj[:, 0], label="X")
    axes[0, 0].plot(t, traj[:, 1], label="Y")
    axes[0, 0].plot(t, traj[:, 2], label="Z")
    axes[0, 0].set_xlabel("时间 [s]")
    axes[0, 0].set_ylabel("位置 [m]")
    axes[0, 0].set_title("位置分量")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(t, r_mag, "b-", linewidth=1.5)
    axes[0, 1].axhline(y=500, color="gray", linestyle="--", label="小行星表面 (R=500m)")
    axes[0, 1].set_xlabel("时间 [s]")
    axes[0, 1].set_ylabel("距中心距离 [m]")
    axes[0, 1].set_title("径向距离")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(t, traj[:, 3], label="Vx")
    axes[1, 0].plot(t, traj[:, 4], label="Vy")
    axes[1, 0].plot(t, traj[:, 5], label="Vz")
    axes[1, 0].plot(t, v_mag, "k-", linewidth=1.5, label="|V|")
    axes[1, 0].set_xlabel("时间 [s]")
    axes[1, 0].set_ylabel("速度 [m/s]")
    axes[1, 0].set_title("速度分量")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(t, mass, "r-", linewidth=1.5)
    axes[1, 1].set_xlabel("时间 [s]")
    axes[1, 1].set_ylabel("质量 [kg]")
    axes[1, 1].set_title("质量变化")
    axes[1, 1].grid(True)

    plt.suptitle("小行星附着状态历史", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_action_history(actions, dt=1.0, t_max=50.0, save_path=None):
    """
    绘制推力指令曲线

    参数
    ----
    actions : list of np.ndarray
        每一步的推力矢量 [Tx, Ty, Tz]
    dt : float
    t_max : float
        最大推力幅值
    save_path : str or None
    """
    acts = np.array(actions)
    t = np.arange(len(acts)) * dt
    mag = np.linalg.norm(acts, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(t, acts[:, 0], label="Tx")
    axes[0].plot(t, acts[:, 1], label="Ty")
    axes[0].plot(t, acts[:, 2], label="Tz")
    axes[0].axhline(y=t_max, color="red", linestyle="--", alpha=0.5, label=f"±T_max ({t_max}N)")
    axes[0].axhline(y=-t_max, color="red", linestyle="--", alpha=0.3)
    axes[0].set_xlabel("时间 [s]")
    axes[0].set_ylabel("推力分量 [N]")
    axes[0].set_title("推力指令分量")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, mag, "b-", linewidth=1.5)
    axes[1].axhline(y=t_max, color="red", linestyle="--", alpha=0.5, label=f"T_max ({t_max}N)")
    axes[1].set_xlabel("时间 [s]")
    axes[1].set_ylabel("推力幅值 [N]")
    axes[1].set_title("推力幅值")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("推力指令历史", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(log_dir="./logs/eval/", save_path=None):
    """
    从 Monitor 日志绘制训练曲线

    参数
    ----
    log_dir : str
        Monitor 日志目录
    save_path : str or None
    """
    import os
    import pandas as pd

    monitor_file = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(monitor_file):
        print(f"未找到评估日志: {monitor_file}")
        print("请先运行训练，或使用 TensorBoard 查看: tensorboard --logdir=./logs/tensorboard/")
        return

    data = np.load(monitor_file)
    timesteps = data["timesteps"]
    results = data["results"]
    ep_lengths = data["ep_lengths"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(timesteps, results.mean(axis=1), "b-", linewidth=1.5)
    axes[0].fill_between(
        timesteps,
        results.mean(axis=1) - results.std(axis=1),
        results.mean(axis=1) + results.std(axis=1),
        alpha=0.2,
    )
    axes[0].set_xlabel("训练步数")
    axes[0].set_ylabel("平均回合奖励")
    axes[0].set_title("评估奖励曲线")
    axes[0].grid(True)

    axes[1].plot(timesteps, ep_lengths.mean(axis=1), "r-", linewidth=1.5)
    axes[1].set_xlabel("训练步数")
    axes[1].set_ylabel("平均回合长度 [步]")
    axes[1].set_title("评估回合长度")
    axes[1].grid(True)

    plt.suptitle("PPO 训练曲线", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
