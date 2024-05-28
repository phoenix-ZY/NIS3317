import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pingouin as pg
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler

# 绘制直方图，显示特征的分布
# 绘制相关性热图，显示特征之间的相关性
# 绘制主成分和因子分析的热图，显示主成分和因子的负荷


def load_data():
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
    return df

def plot_histograms(df):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for i, column in enumerate(df.columns):
        if column != "labels":
            df_sampled = pd.concat(
                [
                    df[df["labels"] == 0].sample(
                        n=2 * min(len(df[df["labels"] == 0]), len(df[df["labels"] == 1]))
                    ),
                    df[df["labels"] == 1].sample(
                        n=min(len(df[df["labels"] == 0]), len(df[df["labels"] == 1]))
                    ),
                ]
            )
            sns.histplot(
                data=df_sampled, x=column, hue="labels", kde=True, ax=axs[i // 4, i % 4]
            )
            axs[i // 4, i % 4].set_title(f"Distribution of {column} by label")
    plt.tight_layout()
    plt.savefig("results/figures/feature_distribution.png")

def plot_correlations(df):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=axs[0])
    axs[0].set_title("Correlation heatmap")
    partial_corrs = []
    for column in df.columns[:-1]:
        partial_corr = pg.partial_corr(
            data=df,
            x=column,
            y="labels",
            covar=df.columns[(df.columns != column) & (df.columns != "labels")].tolist(),
        )
        partial_corrs.append(partial_corr["r"][0])
    sns.barplot(x=df.columns[:-1], y=partial_corrs, ax=axs[1])
    axs[1].set_title("Partial correlation of each feature with the label")
    axs[1].set_ylabel("Partial correlation")
    plt.tight_layout()
    plt.savefig("results/figures/correlation_and_partial_correlation.png")

def perform_pca(df):
    features = df[df.columns[:-1]]
    pca = PCA(n_components=3)
    pca.fit(features)
    components = pca.components_
    components_df = pd.DataFrame(
        components.T, columns=[f"PC{i+1}" for i in range(3)], index=df.columns[:-1]
    )
    return components_df, pca.transform(features)

def perform_fa(df):
    features = df[df.columns[:-1]]
    fa = FactorAnalyzer(n_factors=3, rotation="varimax")
    fa.fit(features)
    loadings = fa.loadings_
    loadings_df = pd.DataFrame(
        loadings, columns=[f"Factor{i+1}" for i in range(3)], index=df.columns[:-1]
    )
    return loadings_df, fa.transform(features)

def plot_heatmaps(components_df, loadings_df):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    sns.heatmap(components_df, annot=True, cmap="coolwarm", ax=axs[0])
    axs[0].set_title("Principal Components Heatmap")
    sns.heatmap(loadings_df, annot=True, cmap="coolwarm", ax=axs[1])
    axs[1].set_title("Factor Loadings Heatmap")
    plt.tight_layout()
    plt.savefig("results/figures/jiangweifenxi.png")

df = load_data()
plot_histograms(df)
plot_correlations(df)
components_df, scores_pca = perform_pca(df)
loadings_df, scores = perform_fa(df)
plot_heatmaps(components_df, loadings_df)