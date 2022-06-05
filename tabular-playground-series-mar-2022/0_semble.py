import warnings
warnings.filterwarnings('ignore')
import random
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import probplot, kurtosis, skew, gmean
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomTreesEmbedding

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, Concatenate, Dropout, BatchNormalization, Activation, GaussianNoise
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


df_train = pd.read_csv('origin/train.csv')
df_test = pd.read_csv('origin/test.csv')

continuous_features = [feature for feature in df_train.columns if feature.startswith('cont')]
target = 'congestion'

print('--------基础分析--------')
print(f'Training Set Shape = {df_train.shape}')
print(f'Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
print(f'Test Set Shape = {df_test.shape}')
print(f'Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

print('--------Target分析--------')
# target is the name of target feature. It follows an extremely left tailed bimodal distribution. Target mean and
# median are very close to each other because there are very few outliers which can be seen from the probability plot.
# Those two outliers are 0 and 3.7, and they should be dealt with.
# target 是目标特征的名称。 它遵循极左尾双峰分布。 目标均值和中位数非常接近，因为从概率图中可以看出很少有异常值。
# 这两个异常值是 0 和 3.7，应该处理它们。

# Bimodal distribution can be break into two components with gaussian mixture model,
# but it is not possible to predict components of test set. At best, it can be done with 61% accuracy,
# which leads models to predict target closer to mean of two components in both training and test set.
# 双峰分布可以用高斯混合模型分解为两个分量，但无法预测测试集的分量。 充其量，它可以以 61% 的准确率完成，
# 这使得模型预测目标更接近训练和测试集中两个组件的平均值。
def plot_target(target):
    df_train[target].astype('float32')
    # 目标特征目标统计分析
    print(f'Target feature {target} Statistical Analysis\n{"-" * 42}')

    print(f'Mean: {df_train[target].mean():.4}  -  Median: {df_train[target].median():.4}  -  Std: {df_train[target].std():.4}')
    print(f'Min: {df_train[target].min()}  -  25%: {df_train[target].quantile(0.25)}  -  50%: {df_train[target].quantile(0.5)}  -  75%: {df_train[target].quantile(0.75)}  -  Max: {df_train[target].max()}')
    # 偏斜 峰度
    print(f'Skew: {df_train[target].skew():.4}  -  Kurtosis: {df_train[target].kurtosis():.4}')
    # 获取null值的个数
    missing_values_count = df_train[df_train[target].isnull()].shape[0]
    # 总数量
    training_samples_count = df_train.shape[0]
    print(f'Missing Values: {missing_values_count}/{training_samples_count} ({missing_values_count * 100 / training_samples_count:.4}%)')

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 12), dpi=100)

    sns.distplot(df_train[target], label=target, ax=axes[0][0])
    axes[0][0].axvline(df_train[target].mean(), label='Target Mean', color='r', linewidth=2, linestyle='--')
    axes[0][0].axvline(df_train[target].median(), label='Target Median', color='b', linewidth=2, linestyle='--')
    probplot(df_train[target], plot=axes[0][1])

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(df_train[target].values.reshape(-1, 1))
    df_train[f'{target}_class'] = gmm.predict(df_train[target].values.reshape(-1, 1))

    sns.distplot(df_train[target], label=target, ax=axes[1][0])
    sns.distplot(df_train[df_train[f'{target}_class'] == 0][target], label='Component 1', ax=axes[1][1])
    sns.distplot(df_train[df_train[f'{target}_class'] == 1][target], label='Component 2', ax=axes[1][1])

    axes[0][0].legend(prop={'size': 15})
    axes[1][1].legend(prop={'size': 15})

    for i in range(2):
        for j in range(2):
            axes[i][j].tick_params(axis='x', labelsize=12)
            axes[i][j].tick_params(axis='y', labelsize=12)
            axes[i][j].set_xlabel('')
            axes[i][j].set_ylabel('')
    axes[0][0].set_title(f'{target} Distribution in Training Set', fontsize=15, pad=12)
    axes[0][1].set_title(f'{target} Probability Plot', fontsize=15, pad=12)
    axes[1][0].set_title(f'{target} Distribution Before GMM', fontsize=15, pad=12)
    axes[1][1].set_title(f'{target} Distribution After GMM', fontsize=15, pad=12)
    plt.show()


plot_target(target)
