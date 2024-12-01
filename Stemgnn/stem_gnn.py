import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime

# root
import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)

from models.StemGNN.handler import train, test
import argparse
import pandas as pd


# ############################################
# import logging
# def setup_logging(log_file_path):
#     """
#     设置日志系统，输出日志到文件和终端
#     """
#     os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # 确保日志目录存在
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         handlers=[
#             logging.FileHandler(log_file_path, mode='a'),  # 追加模式保存日志到文件
#             logging.StreamHandler(sys.stdout)  # 同时输出到终端
#         ]
#     )
#     sys.stdout = open(log_file_path, 'a')  # 重定向标准输出到日志文件
#     sys.stderr = sys.stdout  # 捕获错误输出
    
# # 初始化日志
# log_dir = "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/Stemgnn"  # 日志目录
# log_file = f"{log_dir}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"  # 日志文件路径
# setup_logging(log_file)

# # 调用 handler 方法
# logging.info("Starting training and testing process...")
# train(train_data, valid_data, args, result_train_file)
# test(test_data, args, result_train_file, result_test_file)
# logging.info("Process completed.")
# ############################################


parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='France_processed_0')
parser.add_argument('--window_size', type=int, default=12)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--train_length', type=float, default=7)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=1)
parser.add_argument('--epoch', type=int, default=2) #################
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--multi_layer', type=int, default=5)
# parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation, e.g., cuda:0 or cpu')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)

parser.add_argument('--root_path', type=str, default='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets', help='root path of the data file')
parser.add_argument('--data', type=str, default='Opennem') #################### custom
parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--data_path', type=str, default='France_processed_0.csv', help='data file')
parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
parser.add_argument('--label_len', type=int,  default=12, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--features', type=str, default='MS', ######################
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='Fossil Gas  - Actual Aggregated [MW]', help='target feature in S or MS task') ##########################
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=2, help='train epochs') ######################
#parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
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
# Преобразуем все колонки к числовому типу, заменяя нечисловые значения на NaN
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Удаляем столбцы, которые содержат NaN (то есть нечисловые значения)
df_numeric = df_numeric.dropna(axis=1)

# Преобразуем DataFrame в numpy массив
data = df_numeric.values
args.data_update = data

# split data
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
        test(test_data, args, result_train_file, result_test_file)
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
    print('done')

    
# --device cuda:0
# python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 12 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path  France_processed_0.csv --target "Fossil Gas  - Actual Aggregated [MW]"
# python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 24 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path  France_processed_0.csv --target "Fossil Gas  - Actual Aggregated [MW]"
# python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 12 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path  France_processed_0.csv --target "Fossil Gas  - Actual Aggregated [MW]"
# python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 24 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path  France_processed_0.csv --target "Fossil Gas  - Actual Aggregated [MW]"
# python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 12 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path  France_processed_0.csv --target "Fossil Gas  - Actual Aggregated [MW]"
# python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 24 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path  France_processed_0.csv --target "Fossil Gas  - Actual Aggregated [MW]"
# python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 12 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path  France_processed_0.csv --target "Fossil Gas  - Actual Aggregated [MW]"
# python Stemgnn/stem_gnn.py --train True --evaluate True --dataset France_processed_0 --window_size 24 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cpu --data Opennem --task_name forecasting --data_path  France_processed_0.csv --target "Fossil Gas  - Actual Aggregated [MW]"






# python stemgnn_run.py --train True --evaluate True --dataset Merged_Data_germany --window_size 12 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /Users/wangbo/EnergyTSF-2/datasets/ --device cpu --data custom --task_name forecasting --data_path Merged_Data_germany.csv --target "Day-ahead Price [EUR/MWh]" | tee output_stemgnn_log.txt
# python stem_gnn.py --train True --evaluate True --dataset Merged_Data_germany --window_size 12 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /Users/wangbo/EnergyTSF-2/datasets/ --device cpu --data custom --task_name forecasting --data_path Merged_Data_germany.csv --target "Day-ahead Price [EUR/MWh]" > output_log.txt 2>&1
