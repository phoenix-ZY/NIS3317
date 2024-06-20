import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.final_layer = nn.Linear(256, 1)

    def forward(self, x):
        x1 = self.branch1(x[:, :32])
        x2 = self.branch2(x[:, 32:800])
        x3 = self.branch3(x[:, 800:])
        x = torch.cat([x1, x2,x3], dim=1)
        return self.final_layer(x)  # 添加一个维度

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    # 加载数据
    X = np.load('X.npy')
    y = np.load('y.npy')

    # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train = X_test = X
    y_train = y_test = y
    y_test_origin = y_test
    # 创建归一化器并拟合数据
    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)


    # 转换为torch.Tensor并移动到GPU
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 创建模型、优化器和损失函数
    model = MyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 用于存储每个epoch的训练和测试损失
    train_losses = []
    test_losses = []
    min_test_loss = float('inf')
    # 训练模型
    for epoch in range(1000):  
        model.train()
        train_loss = 0
        update_interval = 100
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}, Train Loss: {train_loss}")
        for i, (batch_X, batch_y) in enumerate(pbar):
            
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i % update_interval == 0 and i > 0:
                pbar.set_description(f"Epoch {epoch+1}, Train Loss: {train_loss / (pbar.n + 1)}")
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            test_loss = criterion(predictions.squeeze(), y_test)
            test_losses.append(test_loss.item())
            print(f'Epoch {epoch+1}, test_Loss: {test_loss.item()}')

        # 每隔50个epoch保存一次模型
        if test_loss.item() < min_test_loss and epoch > 200:
            min_test_loss = test_loss.item()
            torch.save(model.state_dict(), 'best_model_test.pth')

    # 绘制训练和测试的损失曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.savefig("../result/模型拟合/loss曲线TEST.png")


def test():
    X = np.load('X.npy')
    y = np.load('y.npy')

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    y_test_origin = y_test
    # 创建归一化器并拟合数据
    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)


    # 转换为torch.Tensor并移动到GPU
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 创建模型、优化器和损失函数
    model = MyModel().to(device)
    model_weights = torch.load('9_1_best_model.pth')
    model.load_state_dict(model_weights)
    # 预测
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)

    predictions = predictions.cpu().numpy()

    from sklearn.metrics import r2_score
    print(y_test_origin)
    print(predictions)


    r2 = r2_score(y_test_origin, predictions)
    print('R-squared:', r2)

    df = pd.DataFrame({
    'y_test_origin': y_test_origin.flatten(),
    'predictions': predictions.flatten()
})

    # 将数据框保存为csv文件
    df.to_csv('../result/模型拟合/predictions.csv', index=False)


if __name__ == "__main__":
    # train()
    test()