import itertools
import argparse
import math
import time
import torch
import torch.nn as nn
from models.MTGNN.net import gtnet
import numpy as np
from utils.util import *
from models.MTGNN.trainer import Optim
import os
from datetime import datetime

# 超参数网格
embed_sizes = [32, 128]
hidden_sizes = [128]
learning_rates = [0.001, 0.0001]
batch_sizes = [16, 128]
train_epochs = [3]

# 创建唯一的输出目录，以便与之前的文件区分
result_dir = f'output_tune_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 训练与评估函数 (保持与之前相同)
def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = predict.std(axis=0)
    sigma_g = Ytest.std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device)
            tx = X[:, :, id, :]
            ty = Y[:, id]
            output = model(tx, id)
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)
            scale = scale[:, id]
            loss = criterion(output * scale, ty * scale)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
            grad_norm = optim.step()

        if iter % 100 == 0:
            print(f'iter:{iter:3d} | loss: {loss.item() / (output.size(0) * data.m):.3f}')
        iter += 1
    return total_loss / n_samples

# 创建ArgumentParser用于解析参数
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='/Users/wangbo/EnergyTSF-2/datasets/France_processed_0.csv',
                    help='location of the data file') # /Users/wangbo/EnergyTSF-2/datasets/Merged_Data_germany.csv
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=10, help='number of nodes/variables') # 18
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=24*7, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--layers', type=int, default=5, help='number of layers')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')

parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')


args = parser.parse_args()

device = torch.device('cpu')

def main(embed_size, hidden_size, learning_rate, batch_size, train_epochs):
    # 更新超参数
    args.batch_size = batch_size
    args.lr = learning_rate
    args.epochs = train_epochs
    
    Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize)

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels=args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)
    model = model.to(device)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)

    optim = Optim(model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay)

    print('begin training')
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
        val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)

        print(f'Epoch {epoch} completed: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')

    test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
    return val_loss, val_rae, val_corr, test_acc, test_rae, test_corr

if __name__ == "__main__":
    results = []
    for embed_size, hidden_size, lr, batch_size, epoch in itertools.product(embed_sizes, hidden_sizes, learning_rates, batch_sizes, train_epochs):
        print(f'Running with embed_size={embed_size}, hidden_size={hidden_size}, lr={lr}, batch_size={batch_size}, epochs={epoch}')
        
        val_loss, val_rae, val_corr, test_acc, test_rae, test_corr = main(embed_size, hidden_size, lr, batch_size, epoch)
        
        result = {
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'learning_rate': lr,
            'batch_size': batch_size,
            'epochs': epoch,
            'val_loss': val_loss,
            'val_rae': val_rae,
            'val_corr': val_corr,
            'test_acc': test_acc,
            'test_rae': test_rae,
            'test_corr': test_corr
        }
        results.append(result)

    # 保存结果到 CSV 文件
    results_file = os.path.join(result_dir, 'hyperparameter_tuning_results_mtgnn_france.csv')
    import pandas as pd
    pd.DataFrame(results).to_csv(results_file, index=False)

    print(f'Hyperparameter tuning completed. Results saved to {results_file}')
