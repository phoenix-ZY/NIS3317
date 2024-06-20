
import pandas as pd
import ast
import json
from collections import Counter

df = pd.read_csv('ajk.csv')

# 将labels列中的字符串转换为列表
df['labels'] = df['labels'].apply(ast.literal_eval)

# 将direction列的内容添加到labels列中
df['labels'] = df.apply(lambda row: row['labels'] + [row['direction']], axis=1)
df = df.drop(columns=['direction'])

# 剔除有空信息的行
df_cleaned = df.dropna()
df_cleaned = df_cleaned[df_cleaned['labels'] != '[]']

# 提取room_info中的房间数量
df_cleaned[['室', '厅', '卫']] = df_cleaned['room_info'].str.extract(r'(\d+)室(\d+)厅(\d+)卫').astype(int)
df_cleaned = df_cleaned.drop(columns=['room_info'])

# 提取floor中的总楼层数
df_cleaned['总楼层数'] = df_cleaned['floor'].str.extract(r'共(\d+)层').astype(int)

# 根据总楼层数和楼层信息来确定层级，并转换为数字
def determine_level(floor_info):
    total_floors = int(floor_info.split('共')[1].split('层')[0])
    if total_floors < 6:
        return 0  # 低层
    else:
        if '低层' in floor_info:
            return 0  # 低层
        elif '中层' in floor_info:
            return 1  # 中层
        elif '高层' in floor_info:
            return 2  # 高层
        else:
            return -1  # 未知

df_cleaned['层级'] = df_cleaned['floor'].apply(determine_level)

df_cleaned = df_cleaned.drop(columns=['floor'])

# 去掉面积列中的单位，只保留浮点数值
df_cleaned['area'] = df_cleaned['area'].str.replace('㎡', '').astype(float)

df_cleaned = df_cleaned.reset_index(drop=True)
df_cleaned = df_cleaned.drop(index=5)

# 去掉单位价格中的单位，只保留整数数值
df_cleaned['avg_price'] = df_cleaned['avg_price'].str.replace('元/㎡', '').astype(int)

# 去掉history列中的"年建造"字样，只保留整数年份
df_cleaned['history'] = df_cleaned['history'].str.replace('年建造', '').astype(int)

df_cleaned = df_cleaned.drop(columns=['id', 'timestamp'])

# 统计所有标签的出现次数
all_labels = [label for sublist in df_cleaned['labels'] for label in sublist]
label_counts = Counter(all_labels)
label_counts_df = pd.DataFrame(label_counts.items(), columns=['Label', 'Count'])
print(label_counts_df)

# 选择出现频率较高的标签（次数大于100的标签）
selected_labels = label_counts_df[label_counts_df['Count'] > 100]['Label'].tolist()

# 创建标签到数字的映射字典
label_to_number = {label: idx for idx, label in enumerate(selected_labels)}
print(label_to_number)

# 将labels列中的标签转换为数字
df_cleaned['labels'] = df_cleaned['labels'].apply(lambda x: [label_to_number[label] for label in x if label in label_to_number])

# 将字典保存到JSON文件
with open('label_to_number.json', 'w', encoding='utf-8') as f:
    json.dump(label_to_number, f, ensure_ascii=False, indent=4)

df_cleaned = df_cleaned.drop(columns=['price'])

# 创建包含上海所有区的元组
shanghai_districts = (
    "黄浦", "徐汇", "长宁", "静安", "普陀", "虹口", "杨浦", "闵行", "宝山", "嘉定", 
    "浦东", "金山", "松江", "青浦", "奉贤", "崇明"
)

def determine_district(location):
    for district in shanghai_districts:
        if district in location:
            return district
    return "上海周边"

# 创建新的一列 "区"
df_cleaned['区'] = df_cleaned['location'].apply(determine_district)

# 去掉labels列中的重复标签
df_cleaned['labels'] = df_cleaned['labels'].apply(lambda x: list(set(x)))

import pandas as pd

# 假设你的原始数据存储在data中，并且每行数据的标签存储在labels列中
data = pd.read_csv('ajk_cleaned.csv')

import ast


# 定义标签数字到名称的映射
label_mapping = {
    0: "近地铁", 1: "配套成熟", 3: "满五年", 4: "多人关注", 6: "车位充足", 7: "热门小区",
    8: "南北通透", 9: "房东急售", 10: "绿化率高", 11: "户型方正", 12: "随时可看", 13: "唯一住房",
    14: "采光较好", 15: "价格优惠", 17: "有电梯", 19: "次新房", 20: "新上小区", 21: "房东直卖",
    22: "满五", 23: "满二年", 24: "人气热搜", 26: "优势户型", 29: "VIP房东直卖"
}

# 初始化新的列
for label_num, label_name in label_mapping.items():
    data[label_name] = 0

# 填充新的列
for index, row in data.iterrows():
    labels = ast.literal_eval(row['labels'])  # 将字符串形式的列表转换为实际的列表
    for label in labels:
        if label in label_mapping:
            data.at[index, label_mapping[label]] = 1

# 删除原始的labels列
data.drop(columns=['labels'], inplace=True)

# 保存新的数据
data.to_csv('new_data.csv', index=False)

# 保存处理后的数据到新的CSV文件
df_cleaned.to_csv('ajk_cleaned.csv', index=False)

print(df_cleaned.head())
print(df_cleaned.info())
