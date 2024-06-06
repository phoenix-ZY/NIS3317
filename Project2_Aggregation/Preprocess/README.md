# NIS3317

## 数据预处理

`preprocess.py`

数据预处理的过程如下：
- 将不同的指标进行归一化
- 根据不同指标表示的内容，结合实际进行分类，然后对同一类指标的评分值取平均。这里将'art galleries', 'museums', 'resorts', 'parks/picnic spots', 'beaches'归为旅游，'dance clubs', 'juice bars', 'theaters'归为娱乐活动，'restaurants','juice bars'归为饮食，'religious institutions'归为宗教。这里面'juice bars'被归入乐两类，因为其同时具有娱乐和饮食的性质。

`ridarplot.py`

本函数提供了根据四个指标绘制城市雷达图的函数，输入对应的dataframe和行索引，即可绘制出一张四维雷达图。