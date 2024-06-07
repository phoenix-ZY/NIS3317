import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
from ridarplot import plot_radar

global_config = {
    "clusters": 3
}
df = pd.read_csv('data/preprocessed_data.csv')
names = df.iloc[:, 0]
scores = df.iloc[:, 1:]

scaler = StandardScaler()
scores_scaled = scaler.fit_transform(scores)

linked = sch.linkage(scores_scaled, method='ward')
cluster_labels = sch.fcluster(linked, global_config['clusters'], criterion='maxclust')


df['Cluster'] = cluster_labels
print(df)
df.to_csv('result/hierarchical_result.csv', index=False)

df2 = df.iloc[:,1:].groupby('Cluster').mean().reset_index()
print(df2)
plot_radar(df2,[0,1,2], 'result/hierarchical-radarplot.png')



# 可视化示例
plt.figure(figsize=(10, 7))
plt.scatter(scores_scaled[:, 0], scores_scaled[:, 1], c=cluster_labels, cmap='rainbow')
plt.title('聚类结果')
plt.xlabel('指标1')
plt.ylabel('指标2')
plt.show()
plt.savefig("result/hierarchical-clustering.png")