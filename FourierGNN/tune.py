import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from FourierGNN import FGN
import time
import os
import numpy as np
import itertools
import csv
from FourierGNN_utils import save_model, load_model, evaluate
from dataloader import Dataset_FourierGNN


# Hyperparameters grid
#embed_sizes = [32,64,128,256,512]
#hidden_sizes = [32,64,128,256,512]
#learning_rates = [0.001, 0.0001, 0.00001]
#batch_sizes = [2,4,8,16,32,64,128]
#train_epochs = [10]


# Hyperparameters grid
embed_sizes = [32,128,512]
hidden_sizes = [32,128,512]
learning_rates = [0.0001]
batch_sizes = [128]
train_epochs = [10]


parser = argparse.ArgumentParser(description='fourier graph network for multivariate time series forecasting')
parser.add_argument('--data', type=str, default='Merged_Data_germany', help='data set')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
args = parser.parse_args()

csv_file = os.path.join('output', args.data,'hyperparameter_tuning_results.csv')
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['embed_size', 'hidden_size', 'learning_rate', 'batch_size', 'train_loss', 'val_loss'])


# Function to validate the model
def validate(model, vali_loader, forecast_loss, device):
    model.eval()
    cnt = 0
    loss_total = 0
    preds = []
    trues = []
    for i, (x, y) in enumerate(vali_loader):
        cnt += 1
        y = y.float().to(device)
        x = x.float().to(device)
        forecast = model(x)
        y = y.permute(0, 2, 1).contiguous()
        loss = forecast_loss(forecast, y)
        loss_total += float(loss)
        forecast = forecast.detach().cpu().numpy()  # .squeeze()
        y = y.detach().cpu().numpy()  # .squeeze()
        preds.append(forecast)
        trues.append(y)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    score = evaluate(trues, preds)
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    model.train()
    return loss_total / cnt


# Main function
def run_grid_search():
    # Iterate over all hyperparameter combinations
    for embed_size, hidden_size, learning_rate, batch_size, epochs in itertools.product(embed_sizes, hidden_sizes,
                                                                                        learning_rates, batch_sizes,
                                                                                        train_epochs):
        print(
            f"Running for embed_size={embed_size}, hidden_size={hidden_size}, learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")

        # Prepare data
        train_set = Dataset_FourierGNN(root_path='datasets', flag='train', data_path=args.data + '.csv')
        val_set = Dataset_FourierGNN(root_path='datasets', flag='val', data_path=args.data + '.csv')
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize model, optimizer, and loss function
        model = FGN(pre_length=train_set.pred_len, embed_size=embed_size, seq_length=train_set.seq_len,
                    hidden_size=hidden_size).to(device)
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, eps=1e-08)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=0.5)
        forecast_loss = nn.MSELoss(reduction='mean').to(device)

        # Training loop
        for epoch in range(epochs):
            model.train()
            loss_total = 0
            cnt = 0
            for index, (x, y) in enumerate(train_dataloader):
                cnt += 1
                y = y.float().to(device)
                x = x.float().to(device)
                forecast = model(x)
                y = y.permute(0, 2, 1).contiguous()
                loss = forecast_loss(forecast, y)
                loss.backward()
                my_optim.step()
                loss_total += float(loss)

            if (epoch + 1) % 5 == 0:  # Decay every 5 epochs
                my_lr_scheduler.step()

            if (epoch + 1) % 1 == 0:  # Validate every epoch
                val_loss = validate(model, val_dataloader, forecast_loss, device)

            print(f'| end of epoch {epoch + 1} | train_total_loss {loss_total / cnt:.4f} | val_loss {val_loss:.4f}')

        # Log results
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([embed_size, hidden_size, learning_rate, batch_size, loss_total / cnt, val_loss])


if __name__ == '__main__':
    run_grid_search()







# python FourierGNN/tune.py




'''
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from FourierGNN import FGN
import time
import os
import numpy as np
import itertools
import csv
from FourierGNN_utils import save_model, load_model, evaluate
from dataloader import Dataset_FourierGNN

# Hyperparameters grid
embed_sizes = [128]
hidden_sizes = [64]
learning_rates = [0.001] #
batch_sizes = [64] #
train_epochs = [100]


parser = argparse.ArgumentParser(description='Fourier Graph Network for multivariate time series forecasting')
parser.add_argument('--data', type=str, default='Merged_Data_germany', help='data set')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
args = parser.parse_args()

# Create the output directory if it doesn't exist
os.makedirs(os.path.join('output', args.data), exist_ok=True)

# CSV file to save the hyperparameter tuning results
csv_file = os.path.join('output', args.data, 'hyperparameter_tuning_results.csv')
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['embed_size', 'hidden_size', 'learning_rate', 'batch_size', 'train_loss', 'val_loss'])


# Function to validate the model
def validate(model, vali_loader, forecast_loss, device):
    model.eval()
    cnt = 0
    loss_total = 0
    preds = []
    trues = []
    with torch.no_grad():  # Disable gradient calculation for validation
        for x, y in vali_loader:
            cnt += 1
            y = y.float().to(device)
            x = x.float().to(device)
            forecast = model(x)
            y = y.permute(0, 2, 1).contiguous()
            loss = forecast_loss(forecast, y)
            loss_total += float(loss)
            preds.append(forecast.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    score = evaluate(trues, preds)
    print(f'RAW : MAPE {score[0]:.9%}; MAE {score[1]:.9f}; RMSE {score[2]:.9f}.')
    model.train()  # Return to training mode
    return loss_total / cnt


# Main function to run grid search
def run_grid_search():
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Iterate over all hyperparameter combinations
    for embed_size, hidden_size, learning_rate, batch_size, epochs in itertools.product(embed_sizes, hidden_sizes,
                                                                                        learning_rates, batch_sizes,
                                                                                        train_epochs):
        print(f"Running for embed_size={embed_size}, hidden_size={hidden_size}, "
              f"learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")
        
        # Prepare data
        train_set = Dataset_FourierGNN(root_path='datasets', flag='train', data_path=args.data + '.csv')
        val_set = Dataset_FourierGNN(root_path='datasets', flag='val', data_path=args.data + '.csv')
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        # Initialize model, optimizer, and loss function
        model = FGN(pre_length=train_set.pred_len, embed_size=embed_size, seq_length=train_set.seq_len,
                    hidden_size=hidden_size).to(device)
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, eps=1e-08)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)
        forecast_loss = nn.MSELoss(reduction='mean').to(device)

        # Training loop
        for epoch in range(epochs):
            model.train()
            loss_total = 0
            cnt = 0
            start_time = time.time()
            for x, y in train_dataloader:
                cnt += 1
                y = y.float().to(device)
                x = x.float().to(device)
                forecast = model(x)
                y = y.permute(0, 2, 1).contiguous()
                loss = forecast_loss(forecast, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_total += float(loss)

            # Learning rate decay every 5 epochs
            if (epoch + 1) % 5 == 0:
                lr_scheduler.step()

            # Validate the model every epoch
            val_loss = validate(model, val_dataloader, forecast_loss, device)
            elapsed = time.time() - start_time

            print(f'| epoch {epoch + 1:3d} | train_loss {loss_total / cnt:.4f} | val_loss {val_loss:.4f} | '
                  f'time {elapsed:.2f}s')

        # Log results
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([embed_size, hidden_size, learning_rate, batch_size, loss_total / cnt, val_loss])


if __name__ == '__main__':
    run_grid_search()

















import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from FourierGNN import FGN
import time
import os
import numpy as np
import itertools
import csv
from FourierGNN_utils import save_model, load_model, evaluate
from dataloader import Dataset_FourierGNN


# Hyperparameters grid
#embed_sizes = [32,64,128,256,512]
#hidden_sizes = [32,64,128,256,512]
#learning_rates = [0.001, 0.0001, 0.00001]
#batch_sizes = [2,4,8,16,32,64,128]
#train_epochs = [10]


# Hyperparameters grid
embed_sizes = [32,128,512]
hidden_sizes = [32,128,512]
learning_rates = [0.001, 0.0001, 0.00001]
batch_sizes = [2,16,128]
train_epochs = [10]


parser = argparse.ArgumentParser(description='fourier graph network for multivariate time series forecasting')
parser.add_argument('--data', type=str, default='Merged_Data_germany', help='data set')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
args = parser.parse_args()

csv_file = os.path.join('output', args.data,'hyperparameter_tuning_results.csv')
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['embed_size', 'hidden_size', 'learning_rate', 'batch_size', 'train_loss', 'val_loss'])


# Function to validate the model
def validate(model, vali_loader, forecast_loss, device):
    model.eval()
    cnt = 0
    loss_total = 0
    preds = []
    trues = []
    for i, (x, y) in enumerate(vali_loader):
        cnt += 1
        y = y.float().to(device)
        x = x.float().to(device)
        forecast = model(x)
        y = y.permute(0, 2, 1).contiguous()
        loss = forecast_loss(forecast, y)
        loss_total += float(loss)
        forecast = forecast.detach().cpu().numpy()  # .squeeze()
        y = y.detach().cpu().numpy()  # .squeeze()
        preds.append(forecast)
        trues.append(y)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    score = evaluate(trues, preds)
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    model.train()
    return loss_total / cnt


# Main function
def run_grid_search():
    # Iterate over all hyperparameter combinations
    for embed_size, hidden_size, learning_rate, batch_size, epochs in itertools.product(embed_sizes, hidden_sizes,
                                                                                        learning_rates, batch_sizes,
                                                                                        train_epochs):
        print(
            f"Running for embed_size={embed_size}, hidden_size={hidden_size}, learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")

        # Prepare data
        train_set = Dataset_FourierGNN(root_path='datasets', flag='train', data_path=args.data + '.csv')
        val_set = Dataset_FourierGNN(root_path='datasets', flag='val', data_path=args.data + '.csv')
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize model, optimizer, and loss function
        model = FGN(pre_length=train_set.pred_len, embed_size=embed_size, seq_length=train_set.seq_len,
                    hidden_size=hidden_size).to(device)
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, eps=1e-08)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=0.5)
        forecast_loss = nn.MSELoss(reduction='mean').to(device)

        # Training loop
        for epoch in range(epochs):
            model.train()
            loss_total = 0
            cnt = 0
            for index, (x, y) in enumerate(train_dataloader):
                cnt += 1
                y = y.float().to(device)
                x = x.float().to(device)
                forecast = model(x)
                y = y.permute(0, 2, 1).contiguous()
                loss = forecast_loss(forecast, y)
                loss.backward()
                my_optim.step()
                loss_total += float(loss)

            if (epoch + 1) % 5 == 0:  # Decay every 5 epochs
                my_lr_scheduler.step()

            if (epoch + 1) % 1 == 0:  # Validate every epoch
                val_loss = validate(model, val_dataloader, forecast_loss, device)

            print(f'| end of epoch {epoch + 1} | train_total_loss {loss_total / cnt:.4f} | val_loss {val_loss:.4f}')

        # Log results
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([embed_size, hidden_size, learning_rate, batch_size, loss_total / cnt, val_loss])


if __name__ == '__main__':
    run_grid_search()



'''



