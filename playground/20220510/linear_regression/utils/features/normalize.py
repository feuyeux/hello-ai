"""Normalize features"""

import numpy as np

# normalize 归一化的过程是计算:均值/方差 (x-μ)/std
# 均值让数据围绕坐标原点分布
# 除方差让大数变小、整体数值在一个范围内分布


def normalize(features):

    features_normalized = np.copy(features).astype(float)

    # 计算均值
    features_mean = np.mean(features, 0)

    # 计算标准差
    features_deviation = np.std(features, 0)

    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
