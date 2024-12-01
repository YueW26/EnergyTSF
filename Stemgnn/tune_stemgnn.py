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

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, required=False, default=True)
parser.add_argument('--evaluate', type=bool, required=False, default=True)
parser.add_argument('--dataset', type=str, required=False, default='France_processed_0')
parser.add_argument('--window_size', type=int, required=False, default=12)
parser.add_argument('--horizon', type=int, required=False, default=12)
parser.add_argument('--train_length', type=float, required=False, default=7)
parser.add_argument('--valid_length', type=float, required=False, default=2)
parser.add_argument('--test_length', type=float, required=False, default=1)
parser.add_argument('--epoch', type=int, required=False, default=100)
parser.add_argument('--lr', type=float, required=False, default=1e-4)
parser.add_argument('--multi_layer', type=int, required=False, default=5)
# parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--device', type=str, required=False, default='cuda:0', help='Device to use for computation, e.g., cuda:0 or cpu')
parser.add_argument('--validate_freq', type=int, required=False, default=1)
parser.add_argument('--batch_size', type=int, required=False, default=32)
parser.add_argument('--norm_method', type=str, required=False, default='z_score')
parser.add_argument('--optimizer', type=str, required=False, default='RMSProp')
parser.add_argument('--early_stop', type=bool, required=False, default=True)
parser.add_argument('--exponential_decay_step', type=int, required=False, default=5)
parser.add_argument('--decay_rate', type=float, required=False, default=0.5)
parser.add_argument('--dropout_rate', type=float, required=False, default=0.5)
parser.add_argument('--leakyrelu_rate', type=int, required=False, default=0.2)

parser.add_argument('--root_path', type=str, required=False, default='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets', help='root path of the data file')
parser.add_argument('--data', type=str, required=False, default='Opennem') #################### custom
parser.add_argument('--embed', type=str, required=False, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, required=False, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--data_path', type=str, required=False, default='France_processed_0.csv', help='data file')
parser.add_argument('--seq_len', type=int, required=False, default=24, help='input sequence length')
parser.add_argument('--label_len', type=int, required=False, default=12, help='start token length')
parser.add_argument('--pred_len', type=int, required=False, default=24, help='prediction sequence length')
parser.add_argument('--features', type=str, required=False, default='MS', ######################
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, required=False, default='Fossil Gas  - Actual Aggregated [MW]', help='target feature in S or MS task') ##########################
parser.add_argument('--seasonal_patterns', type=str, required=False, default='Monthly', help='subset for M4')
# optimization
parser.add_argument('--num_workers', type=int, required=False, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, required=False, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, required=False, default=10, help='train epochs')
parser.add_argument('--patience', type=int, required=False, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, required=False, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, required=False, default='test', help='exp description')
parser.add_argument('--loss', type=str, required=False, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, required=False, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', required=False, help='use automatic mixed precision training', default=False)
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

    



#####python Stemgnn/tune_stemgnn.py
### python Stemgnn/tune_stemgnn.py --train True --evaluate True --dataset France_processed_0 --window_size 12 --horizon 3 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --root_path /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ --device cuda:0 --data Opennem --task_name forecasting --data_path  France_processed_0.csv --target "Fossil Gas  - Actual Aggregated [MW]"









'''
import os
import torch
from datetime import datetime
from models.StemGNN.handler import train, test
import pandas as pd
import itertools

# Define the hyperparameter grid
embed_sizes = [32, 128]
hidden_sizes = [128]
learning_rates = [0.001, 0.0001]
batch_sizes = [16, 128]
train_epochs = [3]

# File paths
root_path = '/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets'
dataset = 'France_processed_0' #
data_file = os.path.join(root_path, dataset + '.csv')
result_train_file = os.path.join('output', dataset, 'train')
result_test_file = os.path.join('output', dataset, 'test')

# Create result directories if they don't exist
os.makedirs(result_train_file, exist_ok=True)
os.makedirs(result_test_file, exist_ok=True)

# Load and preprocess the dataset
df = pd.read_csv(data_file)
df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
data = df_numeric.values

# Split the dataset into training, validation, and test sets
train_length = 7
valid_length = 2
test_length = 1
train_ratio = train_length / (train_length + valid_length + test_length)
valid_ratio = valid_length / (train_length + valid_length + test_length)
test_ratio = 1 - train_ratio - valid_ratio
train_data = data[:int(train_ratio * len(data))]
valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
test_data = data[int((train_ratio + valid_ratio) * len(data)):]

# Set the device
device = 'cpu'

# Result storage for hyperparameter tuning
results = []

# Grid search over hyperparameters
for embed_size, hidden_size, lr, batch_size, epoch in itertools.product(embed_sizes, hidden_sizes, learning_rates, batch_sizes, train_epochs):
    print(f"Running with embed_size={embed_size}, hidden_size={hidden_size}, lr={lr}, batch_size={batch_size}, epochs={epoch}")
    
    # Set up the argument parameters
    class Args:
        train = True
        evaluate = True
        dataset = 'Merged_Data_germany'
        window_size = 12
        horizon = 3
        train_length = 7
        valid_length = 2
        test_length = 1
        epoch = epoch
        lr = lr
        multi_layer = 5
        device = device
        validate_freq = 1
        batch_size = batch_size
        norm_method = 'z_score'
        optimizer = 'RMSProp'
        early_stop = False
        exponential_decay_step = 5
        decay_rate = 0.5
        dropout_rate = 0.5
        leakyrelu_rate = 0.2
        root_path = root_path
        data = 'custom'
        embed = 'timeF'
        freq = 'h'
        task_name = 'forecasting'
        data_path = 'Merged_Data_germany.csv'
        seq_len = 96
        label_len = 48
        pred_len = 96
        features = 'M'
        target = 'Day-ahead Price [EUR/MWh]'
        num_workers = 0  # 从 args 中读取 num_workers
        itr = 1
        train_epochs = epoch
        patience = 3
        learning_rate = lr
        des = 'test'
        loss = 'MSE'
        lradj = 'type1'
        use_amp = False
    args = Args()

    try:
        # Training process
        before_train = datetime.now().timestamp()
        _, normalize_statistic = train(train_data, valid_data, args, result_train_file)
        after_train = datetime.now().timestamp()
        train_time = (after_train - before_train) / 60
        
        # Evaluation process
        before_evaluation = datetime.now().timestamp()
        test(test_data, args, result_train_file, result_test_file)
        after_evaluation = datetime.now().timestamp()
        eval_time = (after_evaluation - before_evaluation) / 60
        
        # Store the results
        result = {
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'learning_rate': lr,
            'batch_size': batch_size,
            'epochs': epoch,
            'train_time': train_time,
            'eval_time': eval_time
        }
        results.append(result)
        print(f"Completed training and evaluation: {result}")
    
    except KeyboardInterrupt:
        print("Exiting from training early.")
        break
    except Exception as e:
        print(f"Error occurred: {e}")
        continue

# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join('output', 'hyperparameter_tuning_results_stemgnn.csv'), index=False)

print("Hyperparameter tuning complete.")
'''