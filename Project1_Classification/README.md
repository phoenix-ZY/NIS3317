# 分类大作业

## 代码结构
- experiment1.py : 用于寻找最优的分类方法， 寻找范围为决策树，逻辑回归，SVM，随机森林，运行结果保存在results/experiment1文件夹下
- experiment2.py : 用于寻找最优的降维方法， 寻找范围为PCA和因子分析，运行结果保存在results/experiment2文件夹下
- experiemnt3.py : 用于寻找最优的采样方法， 寻找范围为SMOTE和RandomUnderSampler，运行结果保存在results/experiment3文件夹下
- plot.py : 绘制直方图，相关性热图，主成分和因子分析的载荷热图，运行结果保存在results/figures文件夹下

## 实验结论

我们最终采用 逻辑回归，不降维，不采样的方法，这样得到的性能最好；但是考虑到拟合时间和性能的权衡，我们认为可以采用因子分析降维后进行逻辑回归的方法。
