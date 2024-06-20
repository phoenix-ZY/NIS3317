import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns

df_cleaned = pd.read_csv('ajk_cleaned.csv')
df_cleaned['总价'] = df_cleaned['avg_price'] * df_cleaned['area']

# 去除异常值
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df_cleaned = remove_outliers(df_cleaned, 'avg_price')
df_cleaned = remove_outliers(df_cleaned, 'area')
df_cleaned = remove_outliers(df_cleaned, '总价')

zh_font = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')

# 各个区域的二手房数量
plt.figure(figsize=(12, 8))
district_counts = df_cleaned['区'].value_counts()
district_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('区域', fontproperties=zh_font)
plt.ylabel('二手房数量', fontproperties=zh_font)
plt.title('各个区域的二手房数量分布', fontproperties=zh_font)
plt.xticks(rotation=45, ha='right', fontproperties=zh_font)
plt.tight_layout()
plt.show()

# 单位价格
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['avg_price'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('单位价格 (元/㎡)', fontproperties=zh_font)
plt.ylabel('频数', fontproperties=zh_font)
plt.title('单位价格直方图', fontproperties=zh_font)
plt.grid(True)
plt.tight_layout()
plt.show()

# 面积
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['area'], bins=30, color='lightgreen', edgecolor='black')
plt.xlabel('面积 (㎡)', fontproperties=zh_font)
plt.ylabel('频数', fontproperties=zh_font)
plt.title('面积直方图', fontproperties=zh_font)
plt.grid(True)
plt.tight_layout()
plt.show()

# 总价
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['总价'], bins=30, color='salmon', edgecolor='black')
plt.xlabel('总价 (元)', fontproperties=zh_font)
plt.ylabel('频数', fontproperties=zh_font)
plt.title('总价直方图', fontproperties=zh_font)
plt.grid(True)
plt.tight_layout()
plt.show()

# 各区单位价格
districts = df_cleaned['区'].unique()
min_price = df_cleaned['avg_price'].min()
max_price = df_cleaned['avg_price'].max()
bins = 30
bin_edges = np.linspace(min_price, max_price, bins + 1)
max_count = 0
for district in districts:
    subset = df_cleaned[df_cleaned['区'] == district]
    counts, _ = np.histogram(subset['avg_price'], bins=bin_edges)
    max_count = max(max_count, counts.max())

for district in districts:
    plt.figure(figsize=(10, 6))
    subset = df_cleaned[df_cleaned['区'] == district]
    plt.hist(subset['avg_price'], bins=bin_edges, color='skyblue', edgecolor='black')
    plt.xlabel('单位价格 (元/㎡)', fontproperties=zh_font)
    plt.ylabel('频数', fontproperties=zh_font)
    plt.title(f'{district}区的单位价格分布', fontproperties=zh_font)
    plt.ylim(0, max_count)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 各区面积
min_area = df_cleaned['area'].min()
max_area = df_cleaned['area'].max()
bin_edges = np.linspace(min_area, max_area, bins + 1)
max_count = 0
for district in districts:
    subset = df_cleaned[df_cleaned['区'] == district]
    counts, _ = np.histogram(subset['area'], bins=bin_edges)
    max_count = max(max_count, counts.max())

for district in districts:
    plt.figure(figsize=(10, 6))
    subset = df_cleaned[df_cleaned['区'] == district]
    plt.hist(subset['area'], bins=bin_edges, color='lightgreen', edgecolor='black')
    plt.xlabel('面积 (㎡)', fontproperties=zh_font)
    plt.ylabel('频数', fontproperties=zh_font)
    plt.title(f'{district}区的面积分布', fontproperties=zh_font)
    plt.ylim(0, max_count)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 各区总价
min_price = df_cleaned['总价'].min()
max_price = df_cleaned['总价'].max()
bin_edges = np.linspace(min_price, max_price, bins + 1)
max_count = 0
for district in districts:
    subset = df_cleaned[df_cleaned['区'] == district]
    counts, _ = np.histogram(subset['总价'], bins=bin_edges)
    max_count = max(max_count, counts.max())

for district in districts:
    plt.figure(figsize=(10, 6))
    subset = df_cleaned[df_cleaned['区'] == district]
    plt.hist(subset['总价'], bins=bin_edges, color='salmon', edgecolor='black')
    plt.xlabel('总价 (元)', fontproperties=zh_font)
    plt.ylabel('频数', fontproperties=zh_font)
    plt.title(f'{district}区的总价分布', fontproperties=zh_font)
    plt.ylim(0, max_count)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 面积-单价散点图
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['area'], df_cleaned['avg_price'], alpha=0.5, color='blue')
plt.xlabel('面积 (㎡)', fontproperties=zh_font)
plt.ylabel('单位价格 (元/㎡)', fontproperties=zh_font)
plt.title('面积-单价散点图', fontproperties=zh_font)
plt.grid(True)
plt.tight_layout()
plt.show()

# 面积-总价散点图
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['area'], df_cleaned['总价'], alpha=0.5, color='red')
plt.xlabel('面积 (㎡)', fontproperties=zh_font)
plt.ylabel('总价 (元)', fontproperties=zh_font)
plt.title('面积-总价散点图', fontproperties=zh_font)
plt.grid(True)
plt.tight_layout()
plt.show()

# 楼层总价箱线图
plt.figure(figsize=(12, 8))
levels = {0: '低层', 1: '中层', 2: '高层'}
for i, (level, label) in enumerate(levels.items(), 1):
    plt.subplot(1, 3, i)
    df_level = df_cleaned[df_cleaned['层级'] == level]
    plt.boxplot(df_level['总价'], patch_artist=True)
    plt.title(f'{label}总价分布', fontproperties=zh_font)
    plt.ylabel('总价 (元)', fontproperties=zh_font)
    plt.xticks([1], [label], fontproperties=zh_font)
plt.tight_layout()
plt.show()

# 楼层分布饼状图
plt.figure(figsize=(8, 8))
floor_counts = df_cleaned['层级'].value_counts()
floor_labels = {0: '低层', 1: '中层', 2: '高层'}
floor_counts.index = floor_counts.index.map(floor_labels)
plt.pie(floor_counts, labels=floor_counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen', 'salmon'], textprops={'fontproperties': zh_font})
plt.title('楼层分布', fontproperties=zh_font)
plt.axis('equal')
plt.show()

# 各个户型数量分布
df_cleaned['户型'] = df_cleaned['室'].astype(str) + '室' + df_cleaned['厅'].astype(str) + '厅' + df_cleaned['卫'].astype(str) + '卫'
huxing_counts = df_cleaned['户型'].value_counts()
huxing_counts = huxing_counts[huxing_counts > 100]

plt.figure(figsize=(14, 8))
huxing_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('户型', fontproperties=zh_font)
plt.ylabel('数量', fontproperties=zh_font)
plt.title('各个户型的数量分布', fontproperties=zh_font)
plt.xticks(rotation=45, ha='right', fontproperties=zh_font)
plt.tight_layout()
plt.show()

# 各个区的平均单位价格折线图
avg_price_per_district = df_cleaned.groupby('区')['avg_price'].mean().sort_values()

plt.figure(figsize=(20, 10))  
plt.plot(avg_price_per_district.index, avg_price_per_district.values, marker='o', color='skyblue')
plt.xlabel('区', fontproperties=zh_font, fontsize=12)
plt.ylabel('平均单位价格 (元/㎡)', fontproperties=zh_font, fontsize=12)
plt.title('各个区的平均单位价格折线图', fontproperties=zh_font, fontsize=16)
plt.xticks(rotation=45, ha='right', fontproperties=zh_font, fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# 相关系数热力图
columns_of_interest = ['avg_price', 'area', '总价', '室', '厅', '卫', '总楼层数', '层级']
corr_matrix = df_cleaned[columns_of_interest].corr()

plt.rcParams['axes.unicode_minus'] = False  

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('相关系数热力图')
plt.show()
