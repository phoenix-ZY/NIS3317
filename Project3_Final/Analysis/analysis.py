import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib
matplotlib.use('TkAgg')


data = pd.read_csv('../result/模型拟合/predictions.csv')
data.columns = ['y_test_origin', 'predictions']

mae = mean_absolute_error(data['y_test_origin'], data['predictions'])
mse = mean_squared_error(data['y_test_origin'], data['predictions'])
rmse = np.sqrt(mse)
r2 = r2_score(data['y_test_origin'], data['predictions'])

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

abs_residuals = np.abs(data['y_test_origin'] - data['predictions'])

threshold = data['y_test_origin'] * 0.25

within_threshold = abs_residuals <= threshold
proportion_within_threshold = within_threshold.mean()

print(f"Proportion of absolute residuals within threshold of true values: {proportion_within_threshold:.2%}")


plt.figure(figsize=(10, 6))
sns.scatterplot(x='y_test_origin', y='predictions', data=data, alpha=0.3)
plt.plot([data['y_test_origin'].min(), data['y_test_origin'].max()], [data['y_test_origin'].min(), data['y_test_origin'].max()], 'r--')
plt.xlabel('真实价格')
plt.ylabel('预测价格')
plt.title('预测价格与真实价格关系')
plt.show()


residuals = data['y_test_origin'] - data['predictions']
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=200, kde=True)
plt.xlabel('残差')
plt.ylabel('频率')
plt.title('残差分布')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['y_test_origin'], y=residuals, alpha=0.3)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('原始价格')
plt.ylabel('残差')
plt.title('残差与原始价格散点图')
plt.show()

