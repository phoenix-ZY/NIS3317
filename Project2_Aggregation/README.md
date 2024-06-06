# 聚类大作业

## 代码结构
- Analysis文件夹 : 其中analysis.py可以对于数据进行分析，生成相关性热力图，柱状图，偏度和峰度图，生成图片在result文件夹中。
- Preprocess文件夹 : 用于对数据进行预处理，对指标进行降维处理，生成降维后的数据集存储在data/preprocessed_data.csv下面。
- Aggregation文件夹 : 其中提供了三种聚类方法，每种聚类方法在聚类完成后会绘制对应的雷达图和数据分布图展示聚类结果，图片保存在result文件夹中的png中，label存储在result文件夹中的csv文件中。
