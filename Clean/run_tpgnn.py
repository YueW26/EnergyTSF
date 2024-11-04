import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_provider.dataset import STAGNN_stamp_Dataset
from utils.utils import evaluate_metric
from config import DefaultConfig
import models.TPGNN as models  # 确保模型模块路径正确

# 定义 preprocess_data 函数
def preprocess_data(data_path, data_root):
    print("Starting data preprocessing...")
    # 读取 CSV 文件，解析日期列
    data = pd.read_csv(data_path, parse_dates=['date'])
    print("Data loaded successfully. Shape:", data.shape)

    # 生成时间戳
    cycle = 12 * 24  # 每小时 12 个样本，一天 24 小时
    T = data.shape[0]
    time_stamp = np.zeros(T)
    for i, timestamp in enumerate(data['date']):
        minutes = timestamp.hour * 60 + timestamp.minute
        time_stamp[i] = minutes / cycle  # 归一化

    # 保存时间戳文件
    stamp_path = os.path.join(data_root, 'time_stamp.npy')
    np.save(stamp_path, time_stamp)
    print("Timestamp generated and saved to", stamp_path)
    
    # 移除 `date` 列，转换其他列为浮点数
    data = data.drop(columns=['date'])
    data = data.apply(pd.to_numeric, errors='coerce')  # 将非数值转为 NaN
    data = data.fillna(0)  # 用 0 替换 NaN

    # 保存处理后的数据文件
    processed_data_path = os.path.join(data_root, 'processed_data.csv')
    data.to_csv(processed_data_path, index=False, header=False)  # 保存为无标题的文件
    print("Processed data saved to", processed_data_path)
    
    # 获取 n_route（特征数量）
    n_route = data.shape[1]
    return processed_data_path, stamp_path, n_route

def train_model():
    # 设置数据路径和模型配置
    data_path = '/home/kit/aifb/cc7738/scratch/EnergyTSF/datasets/Merged_Data_germany.csv'
    data_root = '/home/kit/aifb/cc7738/scratch/EnergyTSF/datasets/'
    
    # 数据预处理，获取 n_route
    processed_data_path, stamp_path, n_route = preprocess_data(data_path, data_root)
    
    # 设置模型配置参数
    opt = DefaultConfig()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 可以根据需要设置为 'cuda' 或 'cpu'
    opt.n_route = n_route  # 动态设置为数据集的特征数
    opt.n_his = 12  # 历史步长
    opt.n_pred = 12  # 预测步长
    opt.batch_size = 32  # 批大小
    opt.epochs = 10  # 训练轮次，可以根据需要调整
    opt.data_path = processed_data_path  # 使用预处理后的数据路径
    opt.adj_matrix_path = None
    opt.stamp_path = stamp_path
    opt.name = 'TPGNN_Experiment'

    # 设置 `distant_mat` 的默认值为单位矩阵并将其赋给 `opt`
    if not hasattr(opt, 'distant_mat') or opt.distant_mat is None:
        opt.distant_mat = torch.eye(opt.n_route).to(opt.device)  # 使用单位矩阵作为默认邻接矩阵

    # 定义掩码
    enc_spa_mask = torch.ones(1, 1, opt.n_route, opt.n_route).to(opt.device)
    enc_tem_mask = torch.ones(1, 1, opt.n_his, opt.n_his).to(opt.device)
    dec_slf_mask = torch.tril(torch.ones((1, 1, opt.n_pred + 1, opt.n_pred + 1)), diagonal=0).to(opt.device)
    dec_mul_mask = torch.ones(1, 1, opt.n_pred + 1, opt.n_his).to(opt.device)

    # 加载数据集
    print("Loading dataset...")
    train_dataset = STAGNN_stamp_Dataset(opt, train=True, val=False)
    val_dataset = STAGNN_stamp_Dataset(opt, train=False, val=True)
    test_dataset = STAGNN_stamp_Dataset(opt, train=False, val=False)
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size)

    # 模型定义
    print("Initializing model...")
    model_class = getattr(models, opt.model)
    model = model_class(opt, enc_spa_mask, enc_tem_mask, dec_slf_mask, dec_mul_mask).to(opt.device)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.adam['weight_decay'])
    
    # 训练和验证循环
    best_val_loss = float('inf')
    print("Starting training loop...")
    for epoch in range(opt.epochs):
        model.train()
        train_loss = 0.0
        for x, stamp, y in train_loader:
            x, stamp, y = x.to(opt.device), stamp.to(opt.device).long(), y.to(opt.device)  # 将 stamp 转换为 long
            optimizer.zero_grad()
            # print(f"x shape: {x.shape}, stamp shape: {stamp.shape}, y shape: {y.shape}")
            # print(f"x: {x.shape if x is not None else None}, stamp: {stamp.shape if stamp is not None else None}, y: {y.shape if y is not None else None}")
            y_pred = model(x, stamp, y, epoch)

            # 检查 y_pred 是否是 tuple，如果是，取第一个元素
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

            # 确保 y_pred 和 y 的维度一致
            min_pred_len = min(y_pred.shape[2], y.shape[2])
            y_pred = y_pred[:, :, :min_pred_len, :]
            y = y[:, :, :min_pred_len, :]

            # 计算损失
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, stamp, y in val_loader:
                x, stamp, y = x.to(opt.device), stamp.to(opt.device).long(), y.to(opt.device)
                y_pred = model(x, stamp, y, epoch)

                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

                # 确保 y_pred 和 y 的维度一致
                min_pred_len = min(y_pred.shape[2], y.shape[2])
                y_pred = y_pred[:, :, :min_pred_len, :]
                y = y[:, :, :min_pred_len, :]

                val_loss += loss_fn(y_pred, y).item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{opt.epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(data_root, 'best_model.pth'))
            print("Saved Best Model")

    # 测试
    print("Testing best model...")
    model.load_state_dict(torch.load(os.path.join(data_root, 'best_model.pth')))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, stamp, y in test_loader:
            x, stamp, y = x.to(opt.device), stamp.to(opt.device).long(), y.to(opt.device)
            # x = x.type(torch.cuda.FloatTensor)
            # stamp = stamp.type(torch.cuda.LongTensor)
            # y = y.type(torch.cuda.FloatTensor)

            # x = x.repeat(2, 1, 1, 1)
            # stamp = stamp.repeat(2, 1)
            # y = y.repeat(2, 1, 1, 1)
            
            y_pred = model(x, stamp, y, epoch)

            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

            # 确保 y_pred 和 y 的维度一致
            min_pred_len = min(y_pred.shape[2], y.shape[2])
            y_pred = y_pred[:, :, :min_pred_len, :]
            y = y[:, :, :min_pred_len, :]

            test_loss += loss_fn(y_pred, y).item()
    
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

if __name__ == '__main__':
    train_model()
