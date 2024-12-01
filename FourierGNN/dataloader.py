import sys
import os

# 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_provider.data_loader import Dataset_Custom
from utils.timefeatures import time_features


class Dataset_FourierGNN(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='France_processed_0.csv',
                 target='Fossil Gas  - Actual Aggregated [MW]', scale=True, timeenc=0, freq='month', seasonal_patterns=None):
        # 检查 flag 是否有效
        if flag not in ['train', 'val', 'test']:
            raise ValueError("Invalid flag value. Must be 'train', 'val', or 'test'.")
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns)
        self.flag = flag

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), parse_dates=['date'])

        # Debug: 检查是否包含 NaN 值
        if df_raw.isnull().sum().sum() > 0:
            print("Warning: Data contains NaN values. Filling NaN values with forward and backward fill.")
            df_raw.fillna(method='ffill', inplace=True)
            df_raw.fillna(method='bfill', inplace=True)

        # 数据列重新排序
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # 数据集划分
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        # 调整数据集大小以满足 seq_len 和 pred_len 的要求
        if num_train <= self.seq_len + self.pred_len:
            raise ValueError("The training set data is too small to meet the requirements of seq_len and pred_len.")
        if num_vali <= self.seq_len + self.pred_len:
            num_vali = self.seq_len + self.pred_len
            num_train = len(df_raw) - num_vali - num_test
            print("Adjusted validation set size to:", num_vali)
        if num_test <= self.seq_len + self.pred_len:
            num_test = self.seq_len + self.pred_len
            num_train = len(df_raw) - num_vali - num_test
            print("Adjusted test set size to:", num_test)

        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 确保边界合理
        if border1 < 0:
            border1 = 0
        if border2 > len(df_raw):
            border2 = len(df_raw)

        # 数据选择
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 数据标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)

            # Debug: 检查标准化后的数据是否包含 NaN
            if pd.isnull(data).sum().sum() > 0:
                print("Warning: Scaled data contains NaN values. Filling NaN values with forward and backward fill.")
                data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values

        else:
            data = df_data.values

        self.train_data = data[border1s[0]:border2s[0]]
        self.valid_data = data[border1s[1]:border2s[1]]
        self.test_data = data[border1s[2]:border2s[2]]

        # Debug: 输出数据集划分结果
        print(f"Data split: train {len(self.train_data)}, val {len(self.valid_data)}, test {len(self.test_data)}")

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_begin = end
        next_end = next_begin + self.pred_len

        # 根据 flag 选择数据
        if self.flag == 'train':
            if end > len(self.train_data) or next_end > len(self.train_data):
                raise IndexError(f"Index {end} or {next_end} out of bounds for train_data.")
            data = self.train_data[begin:end]
            next_data = self.train_data[next_begin:next_end]
        elif self.flag == 'val':
            if end > len(self.valid_data) or next_end > len(self.valid_data):
                raise IndexError(f"Index {end} or {next_end} out of bounds for valid_data.")
            data = self.valid_data[begin:end]
            next_data = self.valid_data[next_begin:next_end]
        else:
            if end > len(self.test_data) or next_end > len(self.test_data):
                raise IndexError(f"Index {end} or {next_end} out of bounds for test_data.")
            data = self.test_data[begin:end]
            next_data = self.test_data[next_begin:next_end]
        return data, next_data

    def __len__(self):
        # 计算数据长度，确保减去 seq_len 和 pred_len
        if self.flag == 'train':
            return len(self.train_data) - self.seq_len - self.pred_len
        elif self.flag == 'val':
            return len(self.valid_data) - self.seq_len - self.pred_len
        else:
            return len(self.test_data) - self.seq_len - self.pred_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == "__main__":
    dataset = Dataset_FourierGNN(root_path='datasets', flag='train', size=None, features='M', 
                                 data_path='France_processed_0.csv',
                                 target='Fossil Gas  - Actual Aggregated [MW]', scale=True, timeenc=0, freq='month', seasonal_patterns=None)
    print("Dataset length:", len(dataset))
    print("First sample:", dataset[0])






'''
import sys
import os

# 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_provider.data_loader import Dataset_Custom
from utils.timefeatures import time_features


class Dataset_FourierGNN(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='France_processed_0.csv',
                 target='Fossil Gas  - Actual Aggregated [MW]', scale=True, timeenc=0, freq='month', seasonal_patterns=None):
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), parse_dates=['date'])

        # df_raw.columns: ['date', ...(other features), target feature]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # Debug: check for any NaN values
        if df_raw.isnull().sum().sum() > 0:
            print("Warning: Data contains NaN values. Consider preprocessing to handle NaNs.")

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        # Make sure each part has enough data length for training, validation, and testing

        if num_train <= self.seq_len + self.pred_len:
            raise ValueError("The training set data is too small to meet the requirements of seq_len and pred_len.")
        if num_vali <= self.seq_len + self.pred_len:
            num_vali = self.seq_len + self.pred_len  # Adjust validation set size
            num_train = len(df_raw) - num_vali - num_test  # Adjust the training set size
            print("The adjusted validation set size is:", num_vali)
        if num_test <= self.seq_len + self.pred_len:
            num_test = self.seq_len + self.pred_len  # Adjusting the test set size
            num_train = len(df_raw) - num_vali - num_test  # Adjust the training set size
            print("The adjusted test set size is:", num_test)

        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if border1 < 0:
            border1 = 0
        if border2 > len(df_raw):
            border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)

            # Debug: check for NaN values after scaling
            if pd.isnull(data).sum().sum() > 0:
                print("Warning: Scaled data contains NaN values.")
                data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values


        else:
            data = df_data.values

        self.train_data = data[border1:border2]
        self.valid_data = data[border2:border2+num_vali]
        self.test_data = data[border2+ num_vali:border2 + num_vali + num_test]

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_begin = end
        next_end = next_begin + self.pred_len
        if self.flag == 'train':
            data = self.train_data[begin:end]
            next_data = self.train_data[next_begin:next_end]
        elif self.flag == 'val':
            data = self.valid_data[begin:end]
            next_data = self.valid_data[next_begin:next_end]
        else:
            data = self.test_data[begin:end]
            next_data = self.test_data[next_begin:next_end]
        return data, next_data

    def __len__(self):
        # minus the label length
        if self.flag == 'train':
            return len(self.train_data) - self.seq_len - self.pred_len
        elif self.flag == 'val':
            return len(self.valid_data) - self.seq_len - self.pred_len
        else:
            return len(self.test_data) - self.seq_len - self.pred_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == "__main__":
    dataset = Dataset_FourierGNN(root_path='datasets', flag='train', size=None, features='M', data_path='France_processed_0.csv',
                              target='Fossil Gas  - Actual Aggregated [MW]', scale=True, timeenc=0, freq='month', seasonal_patterns=None)
    print(len(dataset))

'''