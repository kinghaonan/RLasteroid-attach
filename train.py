"""
PPO 训练脚本 —— 小行星附着轨迹优化

使用 stable-baselines3 PPO + VecNormalize 训练连续推力着陆策略。
训练日志写入 ./logs/tensorboard/，模型保存至 ./logs/best_model/。
"""
import os
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env.asteroid_env_guided import AsteroidLandingEnvGuided
from config import (
    N_ENVS, N_STEPS, BATCH_SIZE, N_EPOCHS, LEARNING_RATE,
    GAMMA, ENT_COEF, VF_COEF, CLIP_RANGE, GAE_LAMBDA,
    TOTAL_TIMESTEPS, EVAL_FREQ, SAVE_FREQ,
    T_MAX, R_ASTEROID,
)


def make_env():
    env = AsteroidLandingEnvGuided()
    env = Monitor(env)
    return env


def train():
    os.makedirs("./logs/tensorboard", exist_ok=True)
    os.makedirs("./logs/best_model", exist_ok=True)
    os.makedirs("./logs/checkpoints", exist_ok=True)

    # 训练环境（向量化 + 归一化）
    train_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=GAMMA
    )

    # 评估环境（独立实例，共享归一化统计）
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0, gamma=GAMMA
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval/",
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_FREQ // N_ENVS, 1),
        save_path="./logs/checkpoints/",
        name_prefix="ppo_asteroid",
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.ReLU,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        clip_range=CLIP_RANGE,
        gae_lambda=GAE_LAMBDA,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/tensorboard/",
    )

    print(f"开始训练，总步数: {TOTAL_TIMESTEPS}")
    print(f"并行环境数: {N_ENVS}, 每轮步数: {N_STEPS}")
    print(f"小行星半径: {R_ASTEROID}m, 最大推力: {T_MAX}N")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save("./logs/asteroid_landing_ppo_final")
    train_env.save("./logs/vec_normalize.pkl")

    print("训练完成！模型已保存至 ./logs/")


if __name__ == "__main__":
    train()
