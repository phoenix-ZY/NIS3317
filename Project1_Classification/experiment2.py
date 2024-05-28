import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# 本代码用于寻找最优的降维方法， 寻找范围为PCA和因子分析
# 加载数据
df = pd.read_csv("DATASET\RAWDATA_HTRU_2.csv", header=None)
df.columns = [
    "feature1",
    "feature2",
    "feature3",
    "feature4",
    "feature5",
    "feature6",
    "feature7",
    "feature8",
    "labels",
]

# 预处理数据（例如，处理缺失值）
df = df.dropna()

# 提取特征列
features = df[df.columns[:-1]]


from sklearn.decomposition import PCA

# 初始化PCA对象
pca = PCA(n_components=3)

# 拟合数据
pca.fit(features)

# 获取主成分
components = pca.components_

# 创建一个数据框来显示主成分
components_df = pd.DataFrame(
    components.T, columns=[f"PC{i+1}" for i in range(3)], index=df.columns[:-1]
)
# 计算主成分得分
scores_pca = pca.transform(features)

# 将主成分得分转换为数据框
scores_pca_df = pd.DataFrame(scores_pca, columns=[f"PC{i+1}" for i in range(3)])

# 将标签添加到主成分得分数据框
scores_pca_df["labels"] = df["labels"]

# 计算每个主成分得分与标签的相关性
correlations_pca = scores_pca_df.corr()["labels"].drop("labels")


# 初始化因子分析器
fa = FactorAnalyzer(n_factors=3, rotation="varimax")

# 拟合数据
fa.fit(features)

# 获取因子载荷矩阵
loadings = fa.loadings_

# 创建一个数据框来显示因子载荷
loadings_df = pd.DataFrame(
    loadings, columns=[f"Factor{i+1}" for i in range(3)], index=df.columns[:-1]
)

# 计算因子得分
scores = fa.transform(features)

# 将因子得分转换为数据框
scores_df = pd.DataFrame(scores, columns=[f"Factor{i+1}" for i in range(3)])

# 将标签添加到因子得分数据框
scores_df["labels"] = df["labels"]

import json
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

# 定义一个函数来计算指标
def calculate_metrics(y_test, y_pred):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_pred)
    }

# 定义一个函数来进行逻辑回归拟合并计算指标
def fit_and_evaluate(X, y, random_states, model_name, threshold=0.2):
    metrics_sum = {
        "Accuracy": 0,
        "Recall": 0,
        "Precision": 0,
        "AUC": 0,
        "FittingTime": 0
    }

    for random_state in random_states:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        clf = LogisticRegression(max_iter=1000)

        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()
        fitting_time = (end_time - start_time) * 1000
        metrics_sum["FittingTime"] += fitting_time


        start_time = time.time()
        y_pred = (clf.predict_proba(X_test)[:,1] >= threshold).astype(bool)
        metrics = calculate_metrics(y_test, y_pred)
        end_time = time.time()

        for key in metrics_sum:
            if key != "FittingTime":
                metrics_sum[key] += metrics[key]

    # 计算平均指标
    results = {key: val / len(random_states) for key, val in metrics_sum.items()}

    # 保存结果到results/experiment2下的json文件
    if not os.path.exists('results/experiment2'):
        os.makedirs('results/experiment2')

    with open(f'results/experiment2/{model_name}_results.json', 'w') as f:
        json.dump({model_name: results}, f)

# 提取标签列
labels = df["labels"]
random_states = [21, 42, 50]

# 对原始数据进行拟合并计算指标
fit_and_evaluate(features, labels, random_states, 'original')

# 对PCA降维后的数据进行拟合并计算指标
fit_and_evaluate(scores_pca_df.drop("labels", axis=1), labels, random_states, 'pca')

# 对因子分析降维后的数据进行拟合并计算指标
fit_and_evaluate(scores_df.drop("labels", axis=1), labels, random_states, 'factor_analysis')