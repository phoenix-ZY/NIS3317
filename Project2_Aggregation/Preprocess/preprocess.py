import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
# 加载数据
df = pd.read_csv("data\\tripadvisor_review.csv")

# 定义要归一化的列
columns_to_scale = ['art galleries', 'dance clubs', 'juice bars', 'restaurants', 'museums', 'resorts', 'parks/picnic spots', 'beaches', 'theaters', 'religious institutions']

# 创建StandardScaler对象
scaler = StandardScaler()

# 对指定列进行归一化
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# 创建新的特征
df['tourism'] = df[['art galleries', 'museums', 'resorts', 'parks/picnic spots', 'beaches']].mean(axis=1)
df['entertainment'] = df[['dance clubs', 'juice bars', 'theaters']].mean(axis=1)
df['religion'] = df['religious institutions']
df['food and drink'] = df[['restaurants','juice bars']].mean(axis=1)
df = df.drop(columns=columns_to_scale)
df.to_csv("data\\preprocessed_data.csv", index=False)