本仓库实现了一个二手房数据爬取与基于多模态网络的价格预测模型，其中具体代码文件夹的功能如下：

- Analysis:对预测结果进行评估分析。

- CrawlingStuff: 从安居客网站上进行数据爬取。

- data: 存储爬取得到的数据和预处理之后的数据。

- DataProcessing&Visualization: 对爬取得到的数据进行预处理。

- Regression: 对预处理后的数据进行文本特征提取和回归

- result: 存储回归结果，数据分析生成的相关图像等。

运行方式：


在运行之前，需要从`huggingface`上下载`bert-base-chinese`中的模型文件置于`Regression/bert-base-chinese`文件夹中。

```bash
cd CrawlingStuff
python extraction.py  # 数据爬取
cd ..
cd "DataProcessing&Visualization"
python process.py   # 数据预处理
cd ..
cd Regression
python Process.py   # 数据处理（文本特征提取）
python nntrainLinear.py  # 回归拟合
cd ..
cd Analysis
python analysis.py   # 结果分析
```