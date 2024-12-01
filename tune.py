import argparse
import os
import torch
from itertools import product
from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np

import sys #######################
import logging ######################


###############################################################################################################################################


####################################################################################################################################
def setup_logging(log_file_path): ######################
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),  # 使用追加模式 'a'
            logging.StreamHandler(sys.stdout)  # 同时输出到终端
        ]
    )
    # Redirect stdout and stderr to log file
    sys.stdout = open(log_file_path, 'a')
    sys.stderr = sys.stdout

def main():
    # Set up logging ##############
    log_file = "experiment_log.txt"  # 日志文件路径 ##############
    setup_logging(log_file) ##############

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='LTF Experiment')
    
    # # Basic configuration
    # parser.add_argument('--task_name', type=str, default='long_term_forecast', 
    #                     help='Task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    # parser.add_argument('--is_training', type=int, default=1, help='Status: 1 for training, 0 for testing')
    # parser.add_argument('--model_id', type=str, default='test', help='Model ID')
    # parser.add_argument('--model', type=str, default='LightTS', 
    #                     help='Model name, options: [Autoformer, Transformer, TimesNet, DLinear, SCINet, LightTS, PatchTST]')
    # parser.add_argument('--metrics', type=str, nargs='+', default=['MSE', 'MAE', 'MAPE', 'RMSE'], 
    #                     help='Evaluation metrics, options: [MSE, MAE, MAPE, RMSE]')
    # parser.add_argument('--data', type=str, default='Opennem', help='Dataset type')
    # parser.add_argument('--root_path', type=str, default='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets', help='Root path of the data file')
    # parser.add_argument('--data_path', type=str, default='Germany_processed_0.csv', help='Data file path')
    # parser.add_argument('--target', type=str, default='Fossil Gas  - Actual Aggregated [MW]', help='Target feature in S or MS task')
    # parser.add_argument('--pred_len', type=int, default=24, help='Prediction sequence length')
    # parser.add_argument('--freq', type=str, default='h', help='Frequency of the data (e.g., "h" for hourly, "d" for daily)')
    # parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='Seasonal patterns of the data')
    # parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping') ##############

    # # Additional configurations
    # parser.add_argument('--features', type=str, default='M', help='Forecasting task type, options:[M, S, MS]')
    # parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')
    # parser.add_argument('--seq_len', type=int, default=48, help='Input sequence length')
    # parser.add_argument('--label_len', type=int, default=24, help='Start token length')
    # parser.add_argument('--top_k', type=int, default=5, help='For TimesBlock')
    # parser.add_argument('--num_kernels', type=int, default=6, help='For Inception')
    # parser.add_argument('--enc_in', type=int, default=16, help='Encoder input size')
    # parser.add_argument('--dec_in', type=int, default=18, help='Decoder input size')
    # parser.add_argument('--c_out', type=int, default=18, help='Output size')
    # parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    # parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    # parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    # parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    # parser.add_argument('--d_ff', type=int, default=2048, help='Feedforward dimension')
    # parser.add_argument('--moving_avg', type=int, default=25, help='Moving average window size')
    # parser.add_argument('--factor', type=int, default=3, help='Attention factor')
    # parser.add_argument('--distil', action='store_false', default=True, help='Use distilling in encoder')
    # parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    # parser.add_argument('--embed', type=str, default='timeF', help='Time features encoding method')
    # parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    # parser.add_argument('--output_attention', action='store_true', help='Output attention in encoder')
    # parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    # parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')

    # # Experiment control parameters
    # parser.add_argument('--iterations', type=int, default=1, help='Number of experiment iterations')
    # parser.add_argument('--train_epochs', type=int, default=1, help='Number of training epochs')
    # parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    # parser.add_argument('--des', type=str, default='Merged_Data_Transformer', help='Experiment description')

    # # GPU and multi-GPU configuration
    # parser.add_argument('--use_gpu', type=bool, default=torch.cuda.is_available(), help='Use GPU if available')
    # parser.add_argument('--use_multi_gpu', type=bool, default=False, help='Enable multi-GPU support')
    # parser.add_argument('--use_amp', type=bool, default=False, help='Use Automatic Mixed Precision training if True')
    # parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment strategy: type1 or type2')

    # parser.add_argument('--gpu', type=str, default='0', help='gpu index Running on CPU as GPU is unavailable.') #############
    
        # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast', #required=True, ##########
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status') #required=True, 
    parser.add_argument('--model_id', type=str, default='test', help='model id') #required=True, 
    parser.add_argument('--model', type=str, default='SCINet', #required=True, ########## DLinear // SCINet // LightTS // PatchTST /
                        help='model name, options: [DLinear, SCINet, LightTS, PatchTST, TimesNet,]')

    # data loader
    parser.add_argument('--data', type=str, default='Opennem', help='dataset type') #required=True,  #############
    parser.add_argument('--root_path', type=str, default='/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='solar_AL.csv', help='data file') ###### Germany_processed_0.csv // France_processed_0.csv // electricity.csv // solar_AL.csv
    parser.add_argument('--features', type=str, default='MS', #############
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='Plant_1', help='target feature in S or MS task') ######### Fossil Gas  - Actual Aggregated [MW] // E1 // Plant_1
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=24, help='input sequence length') ########## 96
    parser.add_argument('--label_len', type=int, default=12, help='start token length') ########## 48
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length') ######################################## 96 192
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=137, help='encoder input size') ######### 16 // 10 // 321 // 137
    parser.add_argument('--dec_in', type=int, default=137, help='decoder input size')  #########
    parser.add_argument('--c_out', type=int, default=18, help='output size')
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
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience') #####################################################
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
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    
    
    args = parser.parse_args()

    # Hyperparameter grid 
    # pred_len = [96, 192]
    embed_sizes = [128]
    hidden_sizes = [64] # , 128, 256
    
    batch_sizes = [64, 128] # 64, 128
    learning_rates = [0.001, 0.0001] # 0.001, 0.0001
    
    train_epochs = [100]

    # Fix seeds and force CPU usage if necessary
    if not args.use_gpu:
        logging.info("Running on CPU as GPU is unavailable.")
        args.use_multi_gpu = False

    # Select experiment type
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

    # Hyperparameter tuning loop
    for embed, hidden, lr, batch in product(embed_sizes, hidden_sizes, learning_rates, batch_sizes):
        args.embed = embed
        args.d_model = hidden
        args.learning_rate = lr
        args.batch_size = batch
        args.train_epochs = train_epochs[0]

        setting = f"{args.task_name}_{args.model_id}_{args.model}_embed_{embed}_hidden_{hidden}_lr_{lr}_batch_{batch}"
        exp = Exp(args)

        if args.is_training:
            logging.info(f"Starting training: {setting}")
            exp.train(setting)
            logging.info(f"Starting testing: {setting}")
            exp.test(setting)
        else:
            logging.info(f"Starting testing: {setting}")
            exp.test(setting, test=1)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()




# git clone https://github.com/ChenS676/EnergyTSF.git 


# srun -p 4090 --pty --gpus 1 -t 4:00:00 bash -i
# squeue
# conda env list 
# conda activate Energy-TSF

# nvidia-smi

### cd /mnt/webscistorage/cc7738/ws_joella/EnergyTSF
### python run.py
### python tune.py
#### python run.py --task_name long_term_forecast --is_training 1 --model_id test --model DLinear --use_gpu True
# srun -p 4090 --pty --gpus 1 python tune.py
# srun -p 4090 --pty --gpus 1 --time 8:00:00 python tune.py