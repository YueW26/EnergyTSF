import argparse
import os
import torch
from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='DLinear Experiment')

    # Basic configuration
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='Task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='Status: 1 for training, 0 for testing')
    parser.add_argument('--model_id', type=str, default='test', help='Model ID')
    parser.add_argument('--model', type=str, default='LightTS', 
                        help='Model name, options: [Autoformer, Transformer, TimesNet, DLinear]')
    parser.add_argument('--metrics', type=str, nargs='+', default=['MSE', 'MAE', 'MAPE', 'RMSE'], 
                        help='Evaluation metrics, options: [MSE, MAE, MAPE, RMSE]')
    parser.add_argument('--data', type=str, default='Opennem', help='Dataset type')
    parser.add_argument('--root_path', type=str, default='/Users/wangbo/EnergyTSF-3/datasets/', help='Root path of the data file')
    parser.add_argument('--data_path', type=str, default='Germany_processed_0.csv', help='Data file path')  # 添加data_path参数
    parser.add_argument('--target', type=str, default='Fossil Gas  - Actual Aggregated [MW]', help='Target feature in S or MS task')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction sequence length (Horizon set to 24)')
    parser.add_argument('--freq', type=str, default='h', help='Frequency of the data (e.g., "h" for hourly, "d" for daily)')  # 添加freq参数
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='Seasonal patterns of the data (e.g., "Monthly", "Weekly")')  # 添加seasonal_patterns参数
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')  # 添加patience参数

    # Additional configurations
    parser.add_argument('--features', type=str, default='M', help='Forecasting task type, options:[M, S, MS]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')
    parser.add_argument('--seq_len', type=int, default=48, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=24, help='Start token length')
    parser.add_argument('--top_k', type=int, default=5, help='For TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='For Inception')
    parser.add_argument('--enc_in', type=int, default=18, help='Encoder input size')
    parser.add_argument('--dec_in', type=int, default=18, help='Decoder input size')
    parser.add_argument('--c_out', type=int, default=18, help='Output size')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feedforward dimension')
    parser.add_argument('--moving_avg', type=int, default=25, help='Moving average window size')
    parser.add_argument('--factor', type=int, default=3, help='Attention factor')
    parser.add_argument('--distil', action='store_false', default=True, help='Use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--embed', type=str, default='timeF', help='Time features encoding method')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--output_attention', action='store_true', help='Output attention in encoder')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')

    # Experiment control parameters
    parser.add_argument('--iterations', type=int, default=1, help='Number of experiment iterations')
    parser.add_argument('--train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--des', type=str, default='Merged_Data_Transformer', help='Experiment description')  # 添加des参数

    # GPU and multi-GPU configuration
    parser.add_argument('--use_gpu', type=bool, default=torch.cuda.is_available(), help='Use GPU if available')
    parser.add_argument('--use_multi_gpu', type=bool, default=False, help='Enable multi-GPU support')
    parser.add_argument('--use_amp', type=bool, default=False, help='Use Automatic Mixed Precision training if True')
    parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment strategy: type1 or type2')



    try:
        args = parser.parse_args()
    except SystemExit as e:
        print("Argument parsing failed. Please check the required arguments.")
        print(f"SystemExit code: {e}")
        return  # Exit main function gracefully

    # Force CPU usage if GPU is unavailable
    if not args.use_gpu:
        print("Running on CPU as GPU is unavailable.")
        args.use_multi_gpu = False  # Disable multi-GPU if no GPU

    # Select experiment based on task name
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        raise ValueError("Unsupported task name: {}".format(args.task_name))

    # Run experiment
    if args.is_training:
        for ii in range(args.iterations):
            setting = f"{args.task_name}_{args.model_id}_{args.model}_{ii}"
            exp = Exp(args)
            print(f"Starting training: {setting}")
            exp.train(setting)
            print(f"Starting testing: {setting}")
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        setting = f"{args.task_name}_{args.model_id}_{args.model}_0"
        exp = Exp(args)
        print(f"Starting testing: {setting}")
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()



#### python run.py --task_name long_term_forecast --is_training 1 --model_id test --model DLinear


'''
import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
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
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
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
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()        
'''