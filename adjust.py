import os
import torch
import random
import numpy as np
from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

# 设置随机种子
def fix_seed(seed=2021):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

# 超参数网格搜索
def hyperparameter_search(args, param_grid):
    # 使用网格搜索进行超参数调优
    param_grid = ParameterGrid(param_grid)
    best_loss = float('inf')
    best_params = None
    
    for params in param_grid:
        # 更新超参数
        args.learning_rate = params['learning_rate']
        args.batch_size = params['batch_size']
       #args.train_epochs = params['train_epochs']
        
        # 打印当前超参数组合
        print(f"Running with params: {params}")
        
        # 实验设置
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des)

        exp = Exp_Long_Term_Forecast(args)  # 实验初始化
        train_loss = exp.train(setting)  # 训练模型
        
        if train_loss < best_loss:
            best_loss = train_loss
            best_params = params
        
        print(f"Current Loss: {train_loss}, Best Loss: {best_loss}")
    
    return best_params, best_loss

def main():
    fix_seed(2021)

    # 定义argparser模拟输入
    class Args:
        task_name = 'long_term_forecast'
        is_training = 1
        model_id = 'Merged_Data_Transformer'
        model = 'DLinear'
        data = 'custom'
        root_path = '/Users/wangbo/Documents/Time-Series-Library-git/Time-Series-Library-1/datasets'
        data_path = 'Merged_Data_cleaned.csv'
        features = 'M'
        seq_len = 48
        label_len = 24
        pred_len = 24
        enc_in = 18
        dec_in = 18
        c_out = 18
        d_model = 512
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 2048
        factor = 3
        embed = 'timeF'
        distil = True
        des = 'Merged_Data_Transformer'
        itr = 1
        train_epochs = 1
        batch_size = 8
        learning_rate = 1e-4
        num_workers = 8
        target = 'Day-ahead Price [EUR/MWh]'
        dropout = 0.1
        use_gpu = False
        gpu = 0
        use_multi_gpu = False
        devices = '0,1,2,3'
        # 添加 missing 参数 moving_avg
        moving_avg = 25  # 根据实际模型配置选择适当值
        freq = 'h'  # 这里添加 freq 参数
        seasonal_patterns = 'Monthly'  # 添加 seasonal_patterns 参数 可以根据任务需求调整
        checkpoints = './checkpoints/'  # 添加检查点存储路径
        patience = 3  # 添加早停法的 patience 参数
        use_amp = False  # 添加自动混合精度 AMP 参数
        output_attention = False  # 添加是否输出注意力权重的参数
        lradj = 'type1'  # 添加 lradj 参数，默认为 'type1'
     
    args = Args()

    # 定义超参数搜索范围
    param_grid = {
        'learning_rate': [1e-5, 1e-4, 1e-3],
        'batch_size': [8, 16, 32],
        #'train_epochs': [1, 2, 3]
    }
    
    # 执行超参数搜索
    best_params, best_loss = hyperparameter_search(args, param_grid)
    
    # 输出最佳参数
    print(f"Best params: {best_params}")
    print(f"Best loss: {best_loss}")

if __name__ == '__main__':
    main()
