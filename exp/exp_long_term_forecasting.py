from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single

import sys #######################
import logging ######################

warnings.filterwarnings('ignore')

import pandas as pd  ###########################

###########################
def setup_logging(log_file_path): ###########################
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),  # 追加模式
            logging.StreamHandler(sys.stdout)  # 同时输出到终端
        ]
    )
    sys.stdout = open(log_file_path, 'a')  # 重定向标准输出到日志文件
    sys.stderr = sys.stdout  # 捕获错误输出

# class Exp_Long_Term_Forecast(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Long_Term_Forecast, self).__init__(args)
#         self.results = []  ############################
        
class Exp_Long_Term_Forecast(Exp_Basic): #########################
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.results = []
        
        # 设置日志记录文件路径
        log_file = "experiment_long_term_forecast.log"  # 动态生成日志文件名（如需唯一性可加时间戳）
        setup_logging(log_file)  # 调用日志设置函数

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logging.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logging.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            logging.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            logging.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            logging.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        time_start = time.time()  # 开始记录推理时间
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                preds.append(outputs)
                trues.append(batch_y)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        infer_time = time.time() - time_start  # 推理时间
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        logging.info('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        logging.info('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    logging.info("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'not calculated'

        # 初始化 self.results，如果未定义或已被覆盖
        if not hasattr(self, 'results') or not isinstance(self.results, list):
            self.results = []

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        logging.info(f"Test Results - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.6f}")
        logging.info(f"Inference Time: {infer_time:.2f} seconds")

        # 保存实验结果到 self.results
        self.results.append({
            'embed_size': self.args.embed,
            'hidden_size': self.args.d_model,
            'learning_rate': self.args.learning_rate,
            'batch_size': self.args.batch_size,
            'train_epochs': self.args.train_epochs,
            'early_stop_epoch': getattr(self, 'best_epoch', self.args.train_epochs),
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'Train Time (s)': getattr(self, 'total_training_time', 0),
            'Infer Time (s)': infer_time,
        })

        # 创建 "Output" 文件夹（如果不存在）
        output_dir = "Output"
        os.makedirs(output_dir, exist_ok=True)

        # 动态生成唯一的 CSV 文件名
        csv_name = f"results_embed_{self.args.embed}_hidden_{self.args.d_model}_lr_{self.args.learning_rate}_batch_{self.args.batch_size}_epochs_{self.args.train_epochs}.csv"
        csv_name = csv_name.replace(".", "_")  # 替换文件名中的小数点

        # 拼接文件路径
        csv_path = os.path.join(output_dir, csv_name)

        # 保存 CSV 文件到 Output 文件夹
        try:
            pd.DataFrame(self.results).to_csv(csv_path, index=False)
            logging.info(f"Results saved to {csv_path}")
        except Exception as e:
            logging.info(f"Error saving results to CSV: {e}")
            logging.info(f"Type of self.results: {type(self.results)}")
            logging.info(f"Contents of self.results: {self.results}")

        # 保存 NumPy 数据文件
        try:
            np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
            np.save(os.path.join(folder_path, 'pred.npy'), preds)
            np.save(os.path.join(folder_path, 'true.npy'), trues)
            logging.info("Numpy files saved successfully.")
        except Exception as e:
            logging.info(f"Error saving numpy files: {e}")

        return preds, trues
    
    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     if test:
    #         logging.info('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     self.model.eval()
    #     time_start = time.time()  # 开始记录推理时间
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)

    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs[:, -self.args.pred_len:, :]
    #             batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y = batch_y.detach().cpu().numpy()
    #             if test_data.scale and self.args.inverse:
    #                 shape = outputs.shape
    #                 outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
    #                 batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
    #             outputs = outputs[:, :, f_dim:]
    #             batch_y = batch_y[:, :, f_dim:]

    #             pred = outputs
    #             true = batch_y

    #             preds.append(pred)
    #             trues.append(true)
    #             if i % 20 == 0:
    #                 input = batch_x.detach().cpu().numpy()
    #                 if test_data.scale and self.args.inverse:
    #                     shape = input.shape
    #                     input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
    #                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    #                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    #                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    #     infer_time = time.time() - time_start  # 推理时间
    #     preds = np.concatenate(preds, axis=0)
    #     trues = np.concatenate(trues, axis=0)
    #     logging.info('test shape:', preds.shape, trues.shape)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     logging.info('test shape:', preds.shape, trues.shape)

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
        
    #     # dtw calculation
    #     if self.args.use_dtw:
    #         dtw_list = []
    #         manhattan_distance = lambda x, y: np.abs(x - y)
    #         for i in range(preds.shape[0]):
    #             x = preds[i].reshape(-1,1)
    #             y = trues[i].reshape(-1,1)
    #             if i % 100 == 0:
    #                 logging.info("calculating dtw iter:", i)
    #             d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
    #             dtw_list.append(d)
    #         dtw = np.array(dtw_list).mean()
    #     else:
    #         dtw = 'not calculated'


    #     # 假设 self.results 在类的初始化时被定义为一个空列表
    #     self.results = []

    #     mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     logging.info(f"Test Results - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.6f}")
    #     logging.info(f"Inference Time: {infer_time:.2f} seconds")

    #     # 保存实验结果到 self.results
    #     self.results.append({
    #         'embed_size': self.args.embed,
    #         'hidden_size': self.args.d_model,
    #         'learning_rate': self.args.learning_rate,
    #         'batch_size': self.args.batch_size,
    #         'train_epochs': self.args.train_epochs,
    #         'early_stop_epoch': getattr(self, 'best_epoch', self.args.train_epochs),
    #         'MSE': mse,
    #         'MAE': mae,
    #         'MAPE': mape,
    #         'RMSE': rmse,
    #         'Train Time (s)': getattr(self, 'total_training_time', 0),  # 使用类属性
    #         'Infer Time (s)': infer_time,
    #     })

    #     # 创建 "Output" 文件夹（如果不存在）
    #     output_dir = "Output"
    #     os.makedirs(output_dir, exist_ok=True)

    #     # 动态生成唯一的 CSV 文件名
    #     csv_name = f"results_embed_{self.args.embed}_hidden_{self.args.d_model}_lr_{self.args.learning_rate}_batch_{self.args.batch_size}_epochs_{self.args.train_epochs}.csv"
    #     csv_name = csv_name.replace(".", "_")  # 替换文件名中的小数点，防止文件系统不兼容

    #     # 拼接文件路径
    #     csv_path = os.path.join(output_dir, csv_name)

    #     # 保存 CSV 文件到 Output 文件夹
    #     try:
    #         pd.DataFrame(self.results).to_csv(csv_path, index=False)
    #         logging.info(f"Results saved to {csv_path}")
    #     except Exception as e:
    #         logging.info(f"Error saving results to CSV: {e}")

    #     # 保存 NumPy 数据文件
    #     folder_path = "Output/"
    #     os.makedirs(folder_path, exist_ok=True)

    #     try:
    #         np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
    #         np.save(os.path.join(folder_path, 'pred.npy'), preds)
    #         np.save(os.path.join(folder_path, 'true.npy'), trues)
    #         logging.info("Numpy files saved successfully.")
    #     except Exception as e:
    #         logging.info(f"Error saving numpy files: {e}")

    #     return preds, trues
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     if test:
    #         logging.info('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)

    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs[:, -self.args.pred_len:, :]
    #             batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y = batch_y.detach().cpu().numpy()
    #             if test_data.scale and self.args.inverse:
    #                 shape = outputs.shape
    #                 outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
    #                 batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
    #             outputs = outputs[:, :, f_dim:]
    #             batch_y = batch_y[:, :, f_dim:]

    #             pred = outputs
    #             true = batch_y

    #             preds.append(pred)
    #             trues.append(true)
    #             if i % 20 == 0:
    #                 input = batch_x.detach().cpu().numpy()
    #                 if test_data.scale and self.args.inverse:
    #                     shape = input.shape
    #                     input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
    #                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    #                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    #                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    #     preds = np.concatenate(preds, axis=0)
    #     trues = np.concatenate(trues, axis=0)
    #     logging.info('test shape:', preds.shape, trues.shape)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     logging.info('test shape:', preds.shape, trues.shape)

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
        
    #     # dtw calculation
    #     if self.args.use_dtw:
    #         dtw_list = []
    #         manhattan_distance = lambda x, y: np.abs(x - y)
    #         for i in range(preds.shape[0]):
    #             x = preds[i].reshape(-1,1)
    #             y = trues[i].reshape(-1,1)
    #             if i % 100 == 0:
    #                 logging.info("calculating dtw iter:", i)
    #             d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
    #             dtw_list.append(d)
    #         dtw = np.array(dtw_list).mean()
    #     else:
    #         dtw = 'not calculated'
            

    #     mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     logging.info('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
    #     f = open("result_long_term_forecast.txt", 'a')
    #     f.write(setting + "  \n")
    #     f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()

    #     np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #     np.save(folder_path + 'pred.npy', preds)
    #     np.save(folder_path + 'true.npy', trues)

    #     return























'''
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
import pandas as pd  # 用于保存 CSV

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.results = []  # 用于存储每次实验的结果

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            with tqdm(vali_loader, desc="Validation Progress", unit="batch") as tbar:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_dec, batch_x_mark_dec) in enumerate(tbar):
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    batch_x_mark, batch_y_mark = batch_x_mark.to(self.device), batch_y_mark.to(self.device)
                    batch_x_dec, batch_x_mark_dec = batch_x_dec.to(self.device), batch_x_mark_dec.to(self.device)

                    outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                    loss = criterion(outputs, batch_y)
                    total_loss.append(loss.item())
                    tbar.set_postfix(loss=loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        start_time = time.time()
        for epoch in range(self.args.train_epochs):
            epoch_start_time = time.time()
            train_loss = []
            self.model.train()
                    
            with tqdm(train_loader, desc=f"Training Progress (Epoch {epoch + 1}/{self.args.train_epochs})", unit="batch") as tbar:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_dec, batch_x_mark_dec) in enumerate(tbar):
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    batch_x_dec = batch_x_dec.float().to(self.device)
                    batch_x_mark_dec = batch_x_mark_dec.float().to(self.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                            loss = criterion(outputs, batch_y)
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        model_optim.step()

                    train_loss.append(loss.item())
                    tbar.set_postfix(loss=loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            epoch_time = time.time() - epoch_start_time
            logging.info(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Vali Loss: {vali_loss}, Epoch Time: {epoch_time:.2f}s")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        total_training_time = time.time() - start_time  # 训练总时间
        self.total_training_time = total_training_time  # 保存到类属性
        logging.info(f"Total Training Time: {self.total_training_time / 3600:.2f} hours")
        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues = [], []
        infer_start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            with tqdm(test_loader, desc="Testing Progress", unit="batch") as tbar:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_dec, batch_x_mark_dec) in enumerate(tbar):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    batch_x_dec = batch_x_dec.float().to(self.device)
                    batch_x_mark_dec = batch_x_mark_dec.float().to(self.device)

                    outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                    preds.append(outputs.cpu().numpy())
                    trues.append(batch_y.cpu().numpy())
                    tbar.set_postfix(batch=i)

        infer_time = time.time() - infer_start_time
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        logging.info(f"Test Results - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.6f}")
        logging.info(f"Inference Time: {infer_time:.2f} seconds")

        # 保存实验结果到 self.results
        self.results.append({
            'embed_size': self.args.embed,
            'hidden_size': self.args.d_model,
            'learning_rate': self.args.learning_rate,
            'batch_size': self.args.batch_size,
            'train_epochs': self.args.train_epochs,
            'early_stop_epoch': getattr(self, 'best_epoch', self.args.train_epochs),
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'Train Time (s)': getattr(self, 'total_training_time', 0),  # 使用类属性
            'Infer Time (s)': infer_time,
        })





        # 将结果保存为 CSV
        # 创建 "Output" 文件夹（如果不存在）
        output_dir = "Output"
        os.makedirs(output_dir, exist_ok=True)

        # 动态生成唯一的 CSV 文件名
        csv_name = f"results_embed_{self.args.embed}_hidden_{self.args.d_model}_lr_{self.args.learning_rate}_batch_{self.args.batch_size}_epochs_{self.args.train_epochs}.csv"
        csv_name = csv_name.replace(".", "_")  # 替换文件名中的小数点，防止文件系统不兼容

        # 拼接文件路径
        csv_path = os.path.join(output_dir, csv_name)

        # 保存 CSV 文件到 Output 文件夹
        pd.DataFrame(self.results).to_csv(csv_path, index=False)
        logging.info(f"Results saved to {csv_path}")
        
        # 将结果保存为 CSV
        # 动态生成唯一的 CSV 文件名
        # csv_name = f"results_embed_{self.args.embed}_hidden_{self.args.d_model}_lr_{self.args.learning_rate}_batch_{self.args.batch_size}_epochs_{self.args.train_epochs}.csv"
        # csv_name = csv_name.replace(".", "_")  # 替换文件名中的小数点，防止文件系统不兼容
        # pd.DataFrame(self.results).to_csv(csv_name, index=False)
        # logging.info(f"Results saved to {csv_name}")
        return preds, trues

















from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm  #

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            # 添加验证进度条
            with tqdm(vali_loader, desc="Validation Progress", unit="batch") as tbar:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_dec, batch_x_mark_dec) in enumerate(tbar):
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    batch_x_mark, batch_y_mark = batch_x_mark.to(self.device), batch_y_mark.to(self.device)
                    batch_x_dec, batch_x_mark_dec = batch_x_dec.to(self.device), batch_x_mark_dec.to(self.device)

                    outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                    loss = criterion(outputs, batch_y)
                    total_loss.append(loss.item())
                    tbar.set_postfix(loss=loss.item())  # 更新当前 loss
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        start_time = time.time()  # Total training time start
        for epoch in range(self.args.train_epochs):
            epoch_start_time = time.time()  # Per epoch time start
            train_loss = []
            self.model.train()

            # 添加训练进度条
            with tqdm(train_loader, desc=f"Training Progress (Epoch {epoch + 1}/{self.args.train_epochs})", unit="batch") as tbar:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_dec, batch_x_mark_dec) in enumerate(tbar):
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    batch_x_dec = batch_x_dec.float().to(self.device)
                    batch_x_mark_dec = batch_x_mark_dec.float().to(self.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                            loss = criterion(outputs, batch_y)
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        model_optim.step()

                    train_loss.append(loss.item())
                    tbar.set_postfix(loss=loss.item())  # 更新当前 loss

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            epoch_time = time.time() - epoch_start_time  # Time for this epoch
            logging.info(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Vali Loss: {vali_loss}, Epoch Time: {epoch_time:.2f}s")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        total_training_time = time.time() - start_time  # Total training time
        logging.info(f"Total Training Time: {total_training_time / 3600:.2f} hours")
        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

        # Save batch size and training time details
        with open("metrics_log.txt", "a") as f:
            f.write(f"Batch Size: {self.args.batch_size}\n")
            f.write(f"Total Training Time: {total_training_time / 3600:.2f} hours\n")
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues = [], []
        infer_start_time = time.time()  # Inference time start
        self.model.eval()
        with torch.no_grad():
            # 添加测试进度条
            with tqdm(test_loader, desc="Testing Progress", unit="batch") as tbar:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_dec, batch_x_mark_dec) in enumerate(tbar):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    batch_x_dec = batch_x_dec.float().to(self.device)
                    batch_x_mark_dec = batch_x_mark_dec.float().to(self.device)

                    outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                    preds.append(outputs.cpu().numpy())
                    trues.append(batch_y.cpu().numpy())
                    tbar.set_postfix(batch=i)  # 更新当前 batch

        infer_time = time.time() - infer_start_time  # Total inference time
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        logging.info(f"Test Results - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.6f}")
        logging.info(f"Inference Time: {infer_time:.2f} seconds")

        # Save metrics and inference time
        with open("metrics_log.txt", "a") as f:
            f.write(f"Test Results for {setting}:\n")
            f.write(f"MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.6f}\n")
            f.write(f"Inference Time: {infer_time:.2f} seconds\n\n")
        return preds, trues









































from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_dec, batch_x_mark_dec) in enumerate(vali_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_x_mark, batch_y_mark = batch_x_mark.to(self.device), batch_y_mark.to(self.device)
                batch_x_dec, batch_x_mark_dec = batch_x_dec.to(self.device), batch_x_mark_dec.to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        start_time = time.time()  # Total training time start
        for epoch in range(self.args.train_epochs):
            epoch_start_time = time.time()  # Per epoch time start
            train_loss = []
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_dec, batch_x_mark_dec) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x_dec = batch_x_dec.float().to(self.device)
                batch_x_mark_dec = batch_x_mark_dec.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            epoch_time = time.time() - epoch_start_time  # Time for this epoch
            logging.info(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Vali Loss: {vali_loss}, Epoch Time: {epoch_time:.2f}s")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        total_training_time = time.time() - start_time  # Total training time
        logging.info(f"Total Training Time: {total_training_time / 3600:.2f} hours")
        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

        # Save batch size and training time details
        with open("metrics_log.txt", "a") as f:
            f.write(f"Batch Size: {self.args.batch_size}\n")
            f.write(f"Total Training Time: {total_training_time / 3600:.2f} hours\n")
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues = [], []
        infer_start_time = time.time()  # Inference time start
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_dec, batch_x_mark_dec) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x_dec = batch_x_dec.float().to(self.device)
                batch_x_mark_dec = batch_x_mark_dec.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                preds.append(outputs.cpu().numpy())
                trues.append(batch_y.cpu().numpy())

        infer_time = time.time() - infer_start_time  # Total inference time
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        logging.info(f"Test Results - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.6f}")
        logging.info(f"Inference Time: {infer_time:.2f} seconds")

        # Save metrics and inference time
        with open("metrics_log.txt", "a") as f:
            f.write(f"Test Results for {setting}:\n")
            f.write(f"MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.6f}\n")
            f.write(f"Inference Time: {infer_time:.2f} seconds\n\n")
        return preds, trues



























from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_dec, batch_x_mark_dec) in enumerate(vali_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_x_mark, batch_y_mark = batch_x_mark.to(self.device), batch_y_mark.to(self.device)
                batch_x_dec, batch_x_mark_dec = batch_x_dec.to(self.device), batch_x_mark_dec.to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_dec, batch_x_mark_dec) in enumerate(train_loader):
                model_optim.zero_grad()

                # 确保张量类型一致（例如使用 float）并发送到指定设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x_dec = batch_x_dec.float().to(self.device)
                batch_x_mark_dec = batch_x_mark_dec.float().to(self.device)

                # ...

            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_dec, batch_x_mark_dec) in enumerate(train_loader):
            #    model_optim.zero_grad()
            #    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            #    batch_x_mark, batch_y_mark = batch_x_mark.to(self.device), batch_y_mark.to(self.device)
            #    batch_x_dec, batch_x_mark_dec = batch_x_dec.to(self.device), batch_x_mark_dec.to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    loss.backward()
                    model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            logging.info(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Vali Loss: {vali_loss}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_dec, batch_x_mark_dec) in enumerate(test_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_x_mark, batch_y_mark = batch_x_mark.to(self.device), batch_y_mark.to(self.device)
                batch_x_dec, batch_x_mark_dec = batch_x_dec.to(self.device), batch_x_mark_dec.to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_x_dec, batch_x_mark_dec)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                preds.append(outputs.cpu().numpy())
                trues.append(batch_y.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        logging.info(f"Test Results - MSE: {mse}, MAE: {mae}")

        return preds, trues


























# exp_long_term_forecasting.py：主要用于长时间序列预测实验。
# 在长时间预测任务中，模型需要关注更长的时间跨度，
# 因此通常要求较大的 pred_len 和 seq_len。输入数据的长度（seq_len）和预测步长（pred_len）在此文件中通常设置得较大。

# 模块结构较为简单，主要使用了基本的工具函数（如 EarlyStopping 和 adjust_learning_rate），侧重于完成长时间预测实验。

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    #在模型构建时，使用在args中定义的固定配置。
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    # 一般使用均方误差（MSE）作为损失函数，定义了一个简单的 _select_criterion 函数。
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logging.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logging.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            logging.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            logging.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    # vali 函数直接从验证数据中循环处理批次数据进行验证，通过调用模型的 forward 函数获取输出并计算验证误差。
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            logging.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        logging.info('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        logging.info('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        ##########################################
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        #logging.info('mse:{}, mae:{}'.format(mse, mae))
        logging.info('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))
        #logging.info('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        #f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))
        #f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
'''







