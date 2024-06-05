import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

global_config = {
    "clusters": 3
}

df = pd.read_csv('../data/tripadvisor_review.csv')
attractions = df.iloc[:, 0]
data = df.iloc[:, 1:]

gmm = GaussianMixture(n_components=global_config['clusters'], random_state=42)
labels = gmm.fit_predict(data)

df['Cluster'] = labels




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