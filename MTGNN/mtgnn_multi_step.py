import torch
import numpy as np
import argparse
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.util import *
from models.MTGNN.trainer import Trainer
from models.MTGNN.net import gtnet
from data_provider.data_Opennem import Dataset_Opennem
from torch.utils.data import DataLoader


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/France_processed_0.csv', help='data path')
parser.add_argument('--adj_data', type=str, default='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/France_processed_0_adj_mx.pkl', help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False, help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True, help='whether to do curriculum learning')

parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=10, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')

parser.add_argument('--conv_channels', type=int, default=32, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=32, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=64, help='skip channels')
parser.add_argument('--end_channels', type=int, default=128, help='end channels')

parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')

parser.add_argument('--layers', type=int, default=3, help='number of layers')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--step_size1', type=int, default=2500, help='step_size')
parser.add_argument('--step_size2', type=int, default=100, help='step_size')

parser.add_argument('--epochs', type=int, default=100, help='') ##########
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=101, help='random seed')
parser.add_argument('--save', type=str, default='./save/', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='adj alpha')

parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--runs', type=int, default=10, help='number of runs')

args = parser.parse_args()
torch.set_num_threads(3)


def create_dataloader(args, flag):
    dataset = Dataset_Opennem(
        root_path=os.path.dirname(args.data),
        data_path=os.path.basename(args.data),
        size=[args.seq_in_len, args.seq_in_len - 2, args.seq_out_len],
        features='M',
        target='Fossil Gas  - Actual Aggregated [MW]',
        scale=True,
        timeenc=0,
        flag=flag
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(flag == 'train'), drop_last=True)
    return dataloader


def main(runid):
    device = torch.device(args.device)

    train_loader = create_dataloader(args, flag='train')
    val_loader = create_dataloader(args, flag='val')
    test_loader = create_dataloader(args, flag='test')

    dataloader = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': train_loader.dataset.scaler
    }

    adj_data = load_adj(args.adj_data)
    print("Type of adj_data:", type(adj_data))
    print("Content of adj_data:", adj_data)

    predefined_A = adj_data["adj"]
    predefined_A = torch.tensor(predefined_A, dtype=torch.float32) - torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels=args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    print(args)
    print('The receptive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1,
                     args.seq_out_len, dataloader['scaler'], device, args.cl)

    print("Start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    minl = 1e5

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()

        for iter, (x, y, x_mark, y_mark) in enumerate(dataloader['train_loader']):
            trainx = torch.tensor(x, dtype=torch.float32).to(device).unsqueeze(1).transpose(2, 3)
            trainy = torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1).transpose(2, 3)
            trainx = trainx.expand(-1, args.in_dim, -1, -1)

            if iter % args.step_size2 == 0:
                perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes / args.num_split)
            for j in range(args.num_split):
                if j != args.num_split - 1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                id = torch.tensor(id).to(device)
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
                # metrics = engine.train(tx, ty[:, 0, :, :].cpu().numpy(), id)
                metrics = engine.train(tx, ty[:, 0, :, :], id)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y, x_mark, y_mark) in enumerate(dataloader['val_loader']): #####################
            testx = torch.tensor(x, dtype=torch.float32).to(device).unsqueeze(1).transpose(2, 3)
            testy = torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1).transpose(2, 3)
            testx = testx.expand(-1, args.in_dim, -1, -1)

            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)

        if mvalid_loss < minl:
            print(args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth")
            torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth")
            minl = mvalid_loss

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth"))

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    # Test data
    outputs = []
    realy = torch.tensor(dataloader['test_loader'].dataset.data_y, dtype=torch.float32).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y, x_mark, y_mark) in enumerate(dataloader['test_loader']):
        testx = torch.tensor(x, dtype=torch.float32).to(device).unsqueeze(1).transpose(2, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    mae = []
    mape = []
    rmse = []
    for i in range(args.seq_out_len):
        pred = dataloader['scaler'].inverse_transform(yhat[:, :, i].cpu().numpy())
        real = dataloader['scaler'].inverse_transform(realy[:, :, i].cpu().numpy())
        metrics = metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
    return mae, mape, rmse


if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    for i in range(args.runs):
        mae, mape, rmse = main(i)
        vmae.append(mae)
        vmape.append(mape)
        vrmse.append(rmse)

    mae = np.array(vmae)
    mape = np.array(vmape)
    rmse = np.array(vrmse)

    amae = np.mean(mae, 0)
    amape = np.mean(mape, 0)
    armse = np.mean(rmse, 0)

    smae = np.std(mae, 0)
    smape = np.std(mape, 0)
    srmse = np.std(rmse, 0)

    print('\n\nResults for 10 runs\n\n')
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae), np.mean(vrmse), np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae), np.std(vrmse), np.std(vmape)))
    print('\n\n')
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    
    for i in [2, 5, 11]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i + 1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))




'''
import torch
import numpy as np
import argparse
import time

# 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.util import *
from models.MTGNN.trainer import Trainer
from models.MTGNN.net import gtnet


from data_provider.data_Opennem import Dataset_Opennem
from torch.utils.data import DataLoader




def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/France_processed_0.csv',help='data path')

parser.add_argument('--adj_data', type=str,default='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/France_processed_0_adj_mx.pkl',help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=10,help='number of nodes/variables') ########################
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
parser.add_argument('--end_channels',type=int,default=128,help='end channels')


parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')

parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')


parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--save',type=str,default='./save/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')

parser.add_argument('--runs',type=int,default=10,help='number of runs')



args = parser.parse_args()
torch.set_num_threads(3)



def create_dataloader(args, flag):
    """
    创建基于 Dataset_Opennem 的数据加载器。
    """
    dataset = Dataset_Opennem(
        root_path=os.path.dirname(args.data),  # 数据目录（去掉文件名）
        data_path=os.path.basename(args.data),  # 数据文件名
        size=[args.seq_in_len, args.seq_in_len - 2, args.seq_out_len],
        features='M',  # 使用多变量特征
        target='Fossil Gas  - Actual Aggregated [MW]',  # 替换为目标列
        scale=True,
        timeenc=0,  # 时间特征编码
        flag=flag  # 数据集类型（'train', 'val', 'test'）
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(flag == 'train'), drop_last=True)
    return dataloader




def main(runid):
    device = torch.device(args.device)

    # 数据加载
    train_loader = create_dataloader(args, flag='train')
    val_loader = create_dataloader(args, flag='val')
    test_loader = create_dataloader(args, flag='test')

    dataloader = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': train_loader.dataset.scaler
    }

    # 加载邻接矩阵
    adj_data = load_adj(args.adj_data)  # 加载整个字典

    # 调试信息，确保加载的内容正确
    print("Type of adj_data:", type(adj_data))
    print("Content of adj_data:", adj_data)

    # 提取邻接矩阵部分并转换为 PyTorch 张量
    predefined_A = adj_data["adj"]  # 提取邻接矩阵部分
    predefined_A = torch.tensor(predefined_A, dtype=torch.float32) - torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)

    # 初始化模型
    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels=args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    print(args)
    print('The receptive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    # 初始化训练引擎
    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1,
                     args.seq_out_len, dataloader['scaler'], device, args.cl)

    print("Start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    minl = 1e5

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()

        # # 训练循环
        # for iter, (x, y, x_mark, y_mark) in enumerate(dataloader['train_loader']):
        #     trainx = torch.Tensor(x).to(device).transpose(1, 3)
        #     trainy = torch.Tensor(y).to(device).transpose(1, 3)

        #     if iter % args.step_size2 == 0:
        #         perm = np.random.permutation(range(args.num_nodes))
        #     num_sub = int(args.num_nodes / args.num_split)
        #     for j in range(args.num_split):
        #         if j != args.num_split - 1:
        #             id = perm[j * num_sub:(j + 1) * num_sub]
        #         else:
        #             id = perm[j * num_sub:]
        #         id = torch.tensor(id).to(device)
        #         tx = trainx[:, :, id, :]
        #         ty = trainy[:, :, id, :]
        #         metrics = engine.train(tx, ty[:, 0, :, :], id)
        #         train_loss.append(metrics[0])
        #         train_mape.append(metrics[1])
        #         train_rmse.append(metrics[2])

        #     if iter % args.print_every == 0:
        #         log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
        #         print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        # t2 = time.time()
        # train_time.append(t2 - t1)
            
        for iter, (x, y, x_mark, y_mark) in enumerate(dataloader['train_loader']):
            # 检查并调整 x 和 y 的维度
            print(f"Original x shape: {x.shape}, Original y shape: {y.shape}")
            # 将数据转换为 PyTorch 张量并移动到 GPU
            trainx = torch.tensor(x, dtype=torch.float32).to(device).unsqueeze(1).transpose(2, 3)
            trainy = torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1).transpose(2, 3)

            # 将通道扩展到 2
            trainx = trainx.expand(-1, args.in_dim, -1, -1)
            print(f"Processed x shape: {trainx.shape}, Processed y shape: {trainy.shape}")

            
            # 添加额外的通道维度，复制数据，使通道数为 2
            trainx = trainx.expand(-1, args.in_dim, -1, -1)
            print(f"Processed x shape: {trainx.shape}, Processed y shape: {trainy.shape}")

            if iter % args.step_size2 == 0:
                perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes / args.num_split)
            for j in range(args.num_split):
                if j != args.num_split - 1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                id = torch.tensor(id).to(device)
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
                metrics = engine.train(tx, ty[:, 0, :, :], id)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2 - t1)





        # 验证循环
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y, x_mark, y_mark) in enumerate(dataloader['val_loader']):
            testx = torch.Tensor(x).to(device).transpose(1, 3)
            testy = torch.Tensor(y).to(device).transpose(1, 3)

            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        val_time.append(s2 - s1)

        # 记录训练和验证损失
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)

        if mvalid_loss < minl:
            torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth")
            minl = mvalid_loss

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth"))

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    # 测试数据
    outputs = []
    realy = torch.Tensor(dataloader['test_loader'].dataset.data_y).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y, x_mark, y_mark) in enumerate(dataloader['test_loader']):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    mae = []
    mape = []
    rmse = []
    for i in range(args.seq_out_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
    return mae, mape, rmse

if __name__ == "__main__":

    args = parser.parse_args()  ##########################
    
    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    for i in range(args.runs):
        vm1, vm2, vm3, m1, m2, m3 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae,0)
    amape = np.mean(mape,0)
    armse = np.mean(rmse,0)

    smae = np.std(mae,0)
    smape = np.std(mape,0)
    srmse = np.std(rmse,0)

    print('\n\nResults for 10 runs\n\n')
    #valid data
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae),np.mean(vrmse),np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae),np.std(vrmse),np.std(vmape)))
    print('\n\n')
    #test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in [2,5,11]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i+1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))


'''


# python MTGNN/mtgnn_multi_step.py --adj_data /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/France_processed_0_adj_mx.pkl --data /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/France_processed_0.csv --num_nodes 10







# def main(runid):
#     # torch.manual_seed(args.seed)
#     # torch.backends.cudnn.deterministic = True
#     # torch.backends.cudnn.benchmark = False
#     # np.random.seed(args.seed)
#     #load data
#     device = torch.device(args.device)
    

    
#     # dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    
#     train_loader = create_dataloader(args, flag='train')
#     val_loader = create_dataloader(args, flag='val')
#     test_loader = create_dataloader(args, flag='test')

#     dataloader = {
#         'train_loader': train_loader,
#         'val_loader': val_loader,
#         'test_loader': test_loader,
#         'scaler': train_loader.dataset.scaler
#     }

    
#     # 
    
    
#     # Load adjacency matrix
    
#     # Load adjacency matrix
#     adj_data = load_adj(args.adj_data)  # 加载整个字典

#     # 调试信息，确保加载的内容正确
#     print("Type of adj_data:", type(adj_data))
#     print("Content of adj_data:", adj_data)

#     # 提取邻接矩阵部分并转换为 PyTorch 张量
#     predefined_A = adj_data["adj"]  # 提取邻接矩阵部分
#     predefined_A = torch.tensor(predefined_A, dtype=torch.float32) - torch.eye(args.num_nodes)
#     predefined_A = predefined_A.to(device)

#     # 继续模型所需的 scaler
#     scaler = dataloader['scaler']
    
#     # predefined_A = load_adj(args.adj_data)
#     # predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
#     # predefined_A = predefined_A.to(device)   
    




#     # if args.load_static_feature:
#     #     static_feat = load_node_feature('data/sensor_graph/location.csv')
#     # else:
#     #     static_feat = None

#     #model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
#     #               device, predefined_A=predefined_A,
#     #               dropout=args.dropout, subgraph_size=args.subgraph_size,
#     #               node_dim=args.node_dim,
#     #               dilation_exponential=args.dilation_exponential,
#     #               conv_channels=args.conv_channels, residual_channels=args.residual_channels,
#     #               skip_channels=args.skip_channels, end_channels= args.end_channels,
#     #               seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
#     #               layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)
    
#     # # 初始化模型、引擎并进行训练
#     model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
#                   device, predefined_A=None,  # 如果无邻接矩阵，可以使用 None
#                   dropout=args.dropout, subgraph_size=args.subgraph_size,
#                   node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
#                   conv_channels=args.conv_channels, residual_channels=args.residual_channels,
#                   skip_channels=args.skip_channels, end_channels=args.end_channels,
#                   seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
#                   layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)
    
#     print(args)
#     print('The recpetive field size is', model.receptive_field)
#     nParams = sum([p.nelement() for p in model.parameters()])
#     print('Number of model parameters is', nParams)

#     # engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device, args.cl)
#     engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, dataloader['scaler'], device, args.cl)

#     print("start training...",flush=True)
#     his_loss =[]
#     val_time = []
#     train_time = []
#     minl = 1e5
#     for i in range(1,args.epochs+1):
#         train_loss = []
#         train_mape = []
#         train_rmse = []
#         t1 = time.time()
#         # dataloader['train_loader'].shuffle() ###############
#         # for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
#         for iter, (x, y) in enumerate(dataloader['train_loader']): #################
#             trainx = torch.Tensor(x).to(device)
#             trainx= trainx.transpose(1, 3)
#             trainy = torch.Tensor(y).to(device)
#             trainy = trainy.transpose(1, 3)
#             if iter%args.step_size2==0:
#                 perm = np.random.permutation(range(args.num_nodes))
#             num_sub = int(args.num_nodes/args.num_split)
#             for j in range(args.num_split):
#                 if j != args.num_split-1:
#                     id = perm[j * num_sub:(j + 1) * num_sub]
#                 else:
#                     id = perm[j * num_sub:]
#                 id = torch.tensor(id).to(device)
#                 tx = trainx[:, :, id, :]
#                 ty = trainy[:, :, id, :]
#                 metrics = engine.train(tx, ty[:,0,:,:],id)
#                 train_loss.append(metrics[0])
#                 train_mape.append(metrics[1])
#                 train_rmse.append(metrics[2])
#             if iter % args.print_every == 0 :
#                 log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
#                 print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
#         t2 = time.time()
#         train_time.append(t2-t1)
#         #validation
#         valid_loss = []
#         valid_mape = []
#         valid_rmse = []

#         s1 = time.time()
#         for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
#             testx = torch.Tensor(x).to(device)
#             testx = testx.transpose(1, 3)
#             testy = torch.Tensor(y).to(device)
#             testy = testy.transpose(1, 3)
#             metrics = engine.eval(testx, testy[:,0,:,:])
#             valid_loss.append(metrics[0])
#             valid_mape.append(metrics[1])
#             valid_rmse.append(metrics[2])
#         s2 = time.time()
#         log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
#         print(log.format(i,(s2-s1)))
#         val_time.append(s2-s1)
#         mtrain_loss = np.mean(train_loss)
#         mtrain_mape = np.mean(train_mape)
#         mtrain_rmse = np.mean(train_rmse)

#         mvalid_loss = np.mean(valid_loss)
#         mvalid_mape = np.mean(valid_mape)
#         mvalid_rmse = np.mean(valid_rmse)
#         his_loss.append(mvalid_loss)

#         log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
#         print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)

#         if mvalid_loss<minl:
#             torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth")
#             minl = mvalid_loss

#     print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
#     print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


#     bestid = np.argmin(his_loss)
#     engine.model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"))

#     print("Training finished")
#     print("The valid loss on best model is", str(round(his_loss[bestid],4)))

#     #valid data
#     outputs = []
#     realy = torch.Tensor(dataloader['y_val']).to(device)
#     realy = realy.transpose(1,3)[:,0,:,:]

#     for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()): ############# for iter, (x, y) in enumerate(dataloader['train_loader']):
#         testx = torch.Tensor(x).to(device)
#         testx = testx.transpose(1,3)
#         with torch.no_grad():
#             preds = engine.model(testx)
#             preds = preds.transpose(1,3)
#         outputs.append(preds.squeeze())

#     yhat = torch.cat(outputs,dim=0)
#     yhat = yhat[:realy.size(0),...]


#     pred = scaler.inverse_transform(yhat)
#     vmae, vmape, vrmse = metric(pred,realy)

#     #test data
#     outputs = []
#     realy = torch.Tensor(dataloader['y_test']).to(device)
#     realy = realy.transpose(1, 3)[:, 0, :, :]

#     for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
#         testx = torch.Tensor(x).to(device)
#         testx = testx.transpose(1, 3)
#         with torch.no_grad():
#             preds = engine.model(testx)
#             preds = preds.transpose(1, 3)
#         outputs.append(preds.squeeze())

#     yhat = torch.cat(outputs, dim=0)
#     yhat = yhat[:realy.size(0), ...]

#     mae = []
#     mape = []
#     rmse = []
#     for i in range(args.seq_out_len):
#         pred = scaler.inverse_transform(yhat[:, :, i])
#         real = realy[:, :, i]
#         metrics = metric(pred, real)
#         log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
#         print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
#         mae.append(metrics[0])
#         mape.append(metrics[1])
#         rmse.append(metrics[2])
#     return vmae, vmape, vrmse, mae, mape, rmse

