import torch
import torch.nn as nn
import numpy as np
import random
import os
import time
import sys
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import pandas as pd
import logging
from models.TPGNN import TPGNN, predict, predict_stamp
from utils.utils import evaluate_metric
from config import DefaultConfig

# 配置参数
opt = DefaultConfig()
opt.device = 'cpu'  # 使用CPU
opt.data_path = '/Users/wangbo/EnergyTSF-2/datasets/Merged_Data_germany.csv'
opt.n_his = 12  # 历史步长
opt.n_pred = 1  # 预测步长
opt.batch_size = 50  # 批量大小

# 设置日志记录
logging.basicConfig(filename='train_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 自定义数据集类
class MergedDataDataset(Dataset):
    def __init__(self, file_path, seq_length=12, pred_length=1):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.data = pd.read_csv(file_path)

        if 'date' not in self.data.columns:
            raise ValueError("CSV文件中缺少 'date' 列")
        
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)
        self.data = (self.data - self.data.mean()) / self.data.std()

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length

    def __getitem__(self, idx):
        x = self.data.iloc[idx: idx + self.seq_length].values
        y = self.data.iloc[idx + self.seq_length: idx + self.seq_length + self.pred_length].values
        stamp = np.arange(self.pred_length)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # [seq_length, 1, n_attr]
        stamp = torch.tensor(stamp, dtype=torch.long)  # [pred_length]
        y = torch.tensor(y, dtype=torch.float32)

        return x, stamp, y

# 测试函数
def test(model, loss_fn, test_loader, opt):
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for x, stamp, y in test_loader:
            if x.dim() == 3:
                x = x.permute(0, 2, 1)  # [batch, n_route, n_time]
            y_pred = predict_stamp(model, x, stamp, y, opt)
            loss = loss_fn(y_pred, y)
            loss_sum += loss.item()
            n += 1
    return loss_sum / n

# 训练函数
def train(**kwargs):
    opt.parse(kwargs)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)

    batch_size = opt.batch_size
    dataset = MergedDataDataset(opt.data_path, seq_length=opt.n_his, pred_length=opt.n_pred)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)

    # 初始化模型
    model = TPGNN(d_attribute=dataset[0][0].shape[2],  # 输入特征的维度
                  d_out=dataset[0][2].shape[1],  # 输出特征的维度
                  n_route=1,  # 路由数量（根据实际情况调整）
                  n_his=opt.n_his,  # 历史步数
                  dis_mat=None,  # 邻接矩阵（可选）
                  kt=2,  # 时序层的数量
                  n_c=10,  # 另一个参数
                  droprate=0.1,  # dropout 比例
                  temperature=1.0)  # 温度参数

    model.train()
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练过程
    for epoch in range(10):
        for x, stamp, y in train_loader:
            if x.dim() == 3:
                x = x.permute(0, 2, 1)

            print(f"x shape: {x.shape}, stamp shape: {stamp.shape}, y shape: {y.shape}")

            y_pred, loss = model(x, stamp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"y_pred shape: {y_pred.shape}, loss: {loss.item()}")

if __name__ == '__main__':
    import fire
    fire.Fire(train)
