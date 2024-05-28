import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

## 本代码用于寻找最优的分类方法， 寻找范围为决策树，逻辑回归，SVM，随机森林

# 加载数据
data = pd.read_csv('DATASET\RAWDATA_HTRU_2.csv', header=None)

# 分割特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

random_states = [21, 42, 50]
thresholds = np.arange(0.1, 1, 0.1)

results = {}

# 创建一个字典，包含我们要遍历的四种模型
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': svm.SVC(probability=True),
    'Random Forest': RandomForestClassifier()
}

# 在外部添加一个循环，遍历这四种模型
for model_name, model in models.items():
    results = {}

    for threshold in thresholds:
        accuracies = []
        recalls = []
        precisions = []
        auc_rocs = []

        for random_state in random_states:
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

            # 数据标准化
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # 使用训练集对模型进行训练
            model.fit(X_train, y_train)

            # 使用predict_proba方法获取预测概率
            y_pred_proba = model.predict_proba(X_test)

            # 根据预测概率和阈值得到最终的预测结果
            y_pred = (y_pred_proba[:,1] >= threshold).astype('int')

            # 计算并打印评估指标
            accuracies.append(accuracy_score(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred))
            auc_rocs.append(roc_auc_score(y_test, y_pred))

        # 输出平均指标
        results[str(threshold)] = {
            'Average Accuracy': np.mean(accuracies),
            'Average Recall': np.mean(recalls),
            'Average Precision': np.mean(precisions),
            'Average AUC-ROC': np.mean(auc_rocs)
        }

    if not os.path.exists('results/experiment1'):
        os.makedirs('results/experiment1')

    # 保存结果到results下的json文件
    if os.path.exists(f'results/experiment1/{model_name}_results.json') and os.path.getsize(f'results/experiment1/{model_name}_results.json') > 0:
        with open(f'results/experiment1/{model_name}_results.json', 'r+') as f:
            data = json.load(f)
            data[f'{model_name}_stratify'] = results
            f.seek(0)
            f.truncate()
            json.dump(data, f)
    else:
        with open(f'results/experiment1/{model_name}_results.json', 'w') as f:
            json.dump({f'{model_name}_stratify': results}, f)