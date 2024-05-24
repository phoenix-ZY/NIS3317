import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics

# 加载数据
data = pd.read_csv('D:\work\data\htru2\HTRU_2.csv', header=None)

# 分割特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 创建SVM分类器
model = svm.SVC()

# 使用训练集对模型进行训练
model.fit(X_train, y_train)

# 使用测试集进行预测
y_pred = model.predict(X_test)

# 计算准确率
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
