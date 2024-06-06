import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据，跳过第一行
data = pd.read_csv('review.csv')
# 取出B-K列
data = data.iloc[:, 1:11]  # 注意，Python的索引是从0开始的，所以B列是列1，K列是列10

# 计算皮尔森相关系数
pearson_corr = data.corr()

# 使用seaborn的heatmap函数绘制相关系数矩阵的热力图
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', center=0)
plt.show()

# 创建一个新的figure，并设置其大小
plt.figure(figsize=(20, 15))

# 对每个变量进行循环
for i, column in enumerate(data.columns, 1):
    # 创建一个新的subplot
    plt.subplot(4, 3, i)
    # 绘制直方图
    sns.histplot(data[column])
    # 添加标题，显示变量名和方差
    plt.title(f'{column} - Variance: {data[column].var()}')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()

# 计算每个变量的偏度和峰度
skewness = data.skew()
kurtosis = data.kurt()

# 创建一个新的figure，并设置其大小
plt.figure(figsize=(20, 10))

# 绘制偏度的柱状图
plt.subplot(2, 1, 1)
skewness.plot(kind='bar')
plt.title('Skewness')

# 绘制峰度的柱状图
plt.subplot(2, 1, 2)
kurtosis.plot(kind='bar')
plt.title('Kurtosis')

# 显示图形
plt.show()