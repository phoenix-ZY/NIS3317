import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import seaborn as sns
from ridarplot import plot_radar

global_config = {
    "clusters": 3
}

df = pd.read_csv('data/preprocessed_data.csv')
spot_names = df.iloc[:, 0]
data = df.iloc[:, 1:]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=global_config["clusters"], random_state=0)
kmeans.fit(scaled_data)

df['Cluster'] = kmeans.labels_
print(df)
df.to_csv('result/kmeans_result.csv', index=False)

df2 = df.iloc[:,1:].groupby('Cluster').mean().reset_index()
print(df2)
plot_radar(df2,[0,1,2], 'result/kmeans-radarplot.png')

# 可视化示例
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df_pca['Cluster'] = kmeans.labels_

plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='viridis')
plt.title('聚类结果')
plt.show()
plt.savefig("result/kmeans-clustering.png")
