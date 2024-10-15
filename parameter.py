

from models.DLinear import Model
import torch

# 使用之前定义的 Args 类
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

    # 关键的 moving_avg 参数
    moving_avg = 25  # 

    # 其他额外参数
    freq = 'h'
    seasonal_patterns = 'Monthly'
    checkpoints = './checkpoints/'
    patience = 3
    use_amp = False
    output_attention = False
    lradj = 'type1'

# 实例化模型
args = Args()
model = Model(args)

# 打印模型信息
print(model)

# 打印模型参数数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}")
