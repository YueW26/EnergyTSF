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
root_path = '/Users/wangbo/EnergyTSF-2/datasets/'
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
