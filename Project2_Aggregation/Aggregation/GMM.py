import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import seaborn as sns
from ridarplot import plot_radar

global_config = {
    "clusters": 3
}

df = pd.read_csv('data/preprocessed_data.csv')
attractions = df.iloc[:, 0]
data = df.iloc[:, 1:]

gmm = GaussianMixture(n_components=global_config['clusters'], random_state=42)
labels = gmm.fit_predict(data)

df['Cluster'] = labels
df.to_csv('result/gmm_result.csv', index=False)
df2 = df.iloc[:,1:].groupby('Cluster').mean().reset_index()
print(df2)
plot_radar(df2,[0,1,2],savepath='result/gmm-radarplot.png')


# 可视化示例
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data)

df_pca = pd.DataFrame(data=principalComponents, columns=['principal_component_1', 'principal_component_2'])
df_pca['Cluster'] = labels
df_pca['Attraction'] = attractions

plt.figure(figsize=(10, 6))
sns.scatterplot(x='principal_component_1', y='principal_component_2', hue='Cluster', data=df_pca, palette='viridis', s=100)

plt.title('GMM Clustering of Attractions')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
plt.savefig("result/gmm-clustering.png")