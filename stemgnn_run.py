import os
import torch
import pandas as pd
from datetime import datetime
from models.StemGNN.handler import train, test
import argparse
import numpy as np

# 定义评估指标函数
def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

def mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

def mape(predictions, targets):
    return np.mean(np.abs((predictions - targets) / targets)) * 100

def rse(predictions, targets):
    return np.sqrt(np.sum((predictions - targets) ** 2)) / np.sqrt(np.sum((targets - np.mean(targets)) ** 2))

def rae(predictions, targets):
    return np.sum(np.abs(predictions - targets)) / np.sum(np.abs(targets - np.mean(targets)))

# 设置参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='ECG_data')
parser.add_argument('--window_size', type=int, default=12)
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--train_length', type=float, default=7)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=1)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)

parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
# 优化参数
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
args = parser.parse_args()
print(f'Training configs: {args}')

data_file = os.path.join(args.root_path, args.dataset + '.csv')
result_train_file = os.path.join('output', args.dataset, 'train')
result_test_file = os.path.join('output', args.dataset, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)

df = pd.read_csv(data_file)
# 将所有列转换为数值类型，替换非数值值为 NaN
df_numeric = df.apply(pd.to_numeric, errors='coerce')
# 删除包含 NaN 的列
df_numeric = df_numeric.dropna(axis=1)

# 将 DataFrame 转换为 numpy 数组
data = df_numeric.values
args.data_update = data

# 数据切分
train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
test_ratio = 1 - train_ratio - valid_ratio
train_data = data[:int(train_ratio * len(data))]
valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
test_data = data[int((train_ratio + valid_ratio) * len(data)):]

torch.manual_seed(0)
if __name__ == '__main__':
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            _, normalize_statistic = train(train_data, valid_data, args, result_train_file)
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')

    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        predictions, targets = test(test_data, args, result_train_file, result_test_file)

        # 计算原始数据的指标
        raw_rmse = rmse(predictions, targets)
        raw_mae = mae(predictions, targets)
        raw_mape = mape(predictions, targets)
        raw_rse = rse(predictions, targets)
        raw_rae = rae(predictions, targets)

        # 如果需要归一化的数据（假设 `normalize_statistic` 是归一化参数）
        if args.data_update is not None and 'mean' in normalize_statistic and 'scale' in normalize_statistic:
            # 反归一化预测值和真实值
            predictions_norm = predictions * normalize_statistic['scale'] + normalize_statistic['mean']
            targets_norm = targets * normalize_statistic['scale'] + normalize_statistic['mean']

            # 计算归一化数据的指标
            norm_rmse = rmse(predictions_norm, targets_norm)
            norm_mae = mae(predictions_norm, targets_norm)
            norm_mape = mape(predictions_norm, targets_norm)
            norm_rse = rse(predictions_norm, targets_norm)
            norm_rae = rae(predictions_norm, targets_norm)

            print(f'Normalized Metrics: RMSE: {norm_rmse}, MAE: {norm_mae}, MAPE: {norm_mape}, RSE: {norm_rse}, RAE: {norm_rae}')

        # 输出原始数据的指标
        print(f'Raw Metrics: RMSE: {raw_rmse}, MAE: {raw_mae}, MAPE: {raw_mape}, RSE: {raw_rse}, RAE: {raw_rae}')

        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')

    print('done')
