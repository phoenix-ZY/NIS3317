import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from math import pi
# 加载数据
# df = pd.read_csv("data\\preprocessed_data.csv")
categories = ['tourism', 'entertainment', 'religion', 'food and drink']
def plot_radar(df, row_numbers,savepath):
    # 定义数据
    num_vars = len(categories)

    # 计算每个轴的角度
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    # 初始化雷达图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # 为每个变量绘制一个轴并添加标签
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # 绘制y轴标签
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], ["0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4"], color="grey", size=7)
    plt.ylim(0, 1.6)

    # 为每一行数据绘制雷达图
    for row_number in row_numbers:
        values = df.loc[row_number, categories].values.flatten().tolist()
        values = [abs(value) for value in values]  # 对每个值取绝对值
        values += values[:1]  # 重复第一个值以闭合图形

        # 绘制数据
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'Row {row_number}')

        # 填充区域
        ax.fill(angles, values, alpha=0.1)

    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.savefig(savepath)