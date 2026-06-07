"""中心引力场加速度计算 —— 点质量模型"""
import numpy as np
from config import MU


def gravity_accel(r):
    """返回引力加速度矢量 a = -μ·r / |r|³"""
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        return np.zeros(3)
    return -MU * r / r_norm**3
