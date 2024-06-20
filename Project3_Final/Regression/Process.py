import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm 
from sklearn.preprocessing import StandardScaler


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    model = AutoModel.from_pretrained("bert-base-chinese")

    # 加载数据
    data = pd.read_csv('../data/ajk_cleaned_1.csv')
    processed_data = pd.DataFrame()


    # 清洗数据
    data = data.dropna()

    # 将非数值型数据转换为数值型数据
    le = LabelEncoder()
    processed_data['区'] = le.fit_transform(data['区'])
    le = LabelEncoder()
    processed_data['方位'] = le.fit_transform(data['direction'])
    processed_data['面积'] = data['area']
    processed_data['建设时间'] = data['history']
    processed_data['室'] = data['室']
    processed_data['厅'] = data['厅']
    processed_data['卫'] = data['卫']
    processed_data['总楼层数'] = data['总楼层数']
    processed_data['层级'] = data['层级']

    start_col = data.columns.get_loc("近地铁")

    # 选择所有的行和从"近地铁"开始的所有列
    new_df = data.iloc[:, start_col:]

    processed_data = pd.concat([processed_data, new_df], axis=1)

    y = data['avg_price'].values

    x = processed_data.values



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the GPU if available
    model = model.to(device)
    def encode_field(field):
        # Tokenize input
        tokenized_input = tokenizer(field, padding=True, truncation=True, max_length=32, return_tensors='pt')
        tokenized_input = tokenized_input.to(device)

        # Encode input
        with torch.no_grad():
            last_hidden_states = model(**tokenized_input)

        # Get the embeddings of the [CLS] token for each sample
        embeddings = last_hidden_states[0][:,0,:].cpu().numpy()

        return embeddings

    def encode_batches(batched_field):
        embeddings = []
        for batch in tqdm(batched_field, desc='Encoding', total=len(batched_field), leave=False, ncols=80):
            batch_embeddings = encode_field(batch)
            embeddings.append(batch_embeddings)
        return np.concatenate(embeddings)

    # Split 'name' and 'title' columns into batches of size 64
    name_batches = [data['name'].tolist()[i:i + 64] for i in range(0, len(data['name']), 64)]
    print("name_batches prepared")
    title_batches = [data['title'].tolist()[i:i + 64] for i in range(0, len(data['title']), 64)]
    print("title_batches prepared")
    # Encode 'name' and 'title' columns and save them as numpy arrays
    name_encoded = encode_batches(name_batches)
    title_encoded = encode_batches(title_batches)

    x = np.concatenate((x,name_encoded),axis = 1)
    X = np.concatenate((x,title_encoded),axis = 1)

    np.save('X.npy', X)
    np.save('y.npy', y)

    X = np.load('X.npy')
    y = np.load('y.npy')
    print(X.shape)
    print(y.shape)
    




if __name__ == "__main__":
    main()