import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C] # B 是批次大小，T 是时间序列长度，N 是特征数量
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module): # 定义 TimesBlock 类，继承 nn.Module 基类

    # __init__ 是 TimesBlock 类的构造函数，用于初始化 TimesBlock 的所有属性和层。
    # 它负责设置卷积层以及从 configs 参数中读取相关配置。
    # configs 是一个包含各项配置信息的对象或字典。
    def __init__(self, configs): # 定义构造函数，接收一个配置对象 configs 作为参数
        super(TimesBlock, self).__init__() # 调用父类 nn.Module 的初始化方法，以便利用它的功能
        self.seq_len = configs.seq_len # 将 configs 中的序列长度赋值给 self.seq_len
        self.pred_len = configs.pred_len # 将预测长度赋值给 self.pred_len
        self.k = configs.top_k # 配置参数中表示要选择的频率数，赋值给 self.k
        # parameter-efficient design

        # 定义 self.conv，这是一个包含卷积和激活层的组合，用于处理二维卷积操作
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    # forward 方法定义了数据在 TimesBlock 中的流动和操作。
    # 它接收一个输入张量 x，并依次对其执行 FFT 分解、二维卷积和加权叠加，最后输出处理后的张量
    def forward(self, x):
        B, T, N = x.size() # [B, T, C] # B 是批次大小，T 是时间序列长度，N 是特征数量
        period_list, period_weight = FFT_for_Period(x, self.k) # 获取主要频率及其权重

        # 循环每个主要频率，并处理数据
        res = [] # 定义空列表 res，用于存储每个频率的卷积结果
        for i in range(self.k): # 循环 self.k 次，即遍历 top_k 主要频率
            period = period_list[i] # 当前频率的周期
            
            # padding
            # 处理和卷积
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            
            # reshape 
            # 重塑和卷积处理
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)

            # reshape back 
            # 重新整形和残差连接
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        # 权重加权与残差连接    
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
        # 最后，返回处理后的张量 res，这就是 TimesBlock 的输出。


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 这是 forecast 方法的定义。
        # self 代表类的实例本身。
        # x_enc 是输入的时间序列数据，
        # x_mark_enc 是输入数据对应的时间标记，
        # x_dec 是解码器的输入数据（对预测步长有帮助），
        # x_mark_dec 是解码器数据对应的时间标记。
        # Normalization from Non-stationary Transformer

        means = x_enc.mean(1, keepdim=True).detach()
        #[B,T]
        # %对输入数据 x_enc 沿着时间维度（第二个维度，即 T 维度）求平均值。
        # x_enc.mean(1, keepdim=True) 计算每个时间序列的平均值，
        # keepdim=True 保证结果的维度与输入相同。
        # .detach() 表示该操作不参与梯度计算（即不会影响模型训练），用于保持稳定性。
        # means 是一个形状为 [B, 1] 的张量，其中 B 是批次大小，代表不同数据样本的个数。

        x_enc = x_enc - means
        # %将输入的时间序列 x_enc 减去它的均值 means，完成零均值化操作，消除数据的偏移量，使时间序列数据的均值为0。这样可以提高模型的训练效果。

        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # %计算输入数据 x_enc 的标准差（即数据的散布程度）。
        # 首先 torch.var(x_enc, dim=1, keepdim=True, unbiased=False) 计算 x_enc 的方差（在时间维度上），并保持维度不变。
        # 然后通过 torch.sqrt() 计算标准差，并加上一个非常小的数 1e-5，以避免后续计算中出现除零错误。stdev 是一个形状为 [B, 1] 的张量。

        x_enc /= stdev
        # %将 x_enc 除以其标准差 stdev，完成标准化操作。
        # 标准化后的数据平均值为0，标准差为1，有助于提高模型的训练速度和效果。

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # %将输入数据 x_enc 和时间标记 x_mark_enc 通过 self.enc_embedding 方法进行嵌入。
        # enc_embedding 是一种编码方式，将每个时间点的数值数据投影到一个 C 维的向量空间。
        # 最终得到 enc_out，其形状为 [B, T, C]，其中 B 是批次大小，T 是时间序列的长度，C（即 d_model）是每个时间点数据嵌入后的维度。

        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # %首先对 enc_out 的维度进行转换。
        # enc_out.permute(0, 2, 1) 将原来的 [B, T, C] 变为 [B, C, T]，以便能够通过 predict_linear 进行预测步长的操作。
        # self.predict_linear 是一个线性层，预测给定的时间序列后续的步长数据，生成一个形状为 [B, pred_len+seq_len, C] 的张量。
        # 最后，通过 permute(0, 2, 1) 再次调整维度，以对齐时间维度。

        # TimesNet: pass through TimesBlock for self.layer times each with layer normalization
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # %在一个循环中多次将 enc_out 传递给 TimesBlock 层（self.model[i]）。
        # 每次经过 TimesBlock 后，还会对结果进行层归一化（self.layer_norm）。TimesBlock 是 TimesNet 的核心模块，用于提取数据的时间周期信息。
        # 这一过程会重复 self.layer 次（即模型的层数）。

        # porject back
        dec_out = self.projection(enc_out)
        # %将 enc_out 通过线性投影层 self.projection 映射回原始输出空间。
        # 结果 dec_out 的形状为 [B, T, c_out]，其中 c_out 是最终输出的维度，通常用于指定模型输出的特征数量。

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        # %对 dec_out 进行反标准化处理。
        # 首先 stdev[:, 0, :] 取出每个数据样本的标准差，并使用 .unsqueeze(1) 增加一个维度，
        # 然后通过 .repeat 函数复制，使其维度与 dec_out 匹配。这样做的目的是将预测数据还原到初始数据的尺度。

        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        # %对 dec_out 进行偏移还原（即将之前去除的均值加回来）。
        # 同样地，将 means[:, 0, :] 复制到与 dec_out 相同的维度上，然后加到 dec_out 中。
        # 此操作完成后，预测结果的尺度就与原始数据一致了。
        return dec_out
        # 最后，返回最终的预测结果 dec_out。#

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    # 这是 forward 方法的定义。该方法接收五个参数，self 表示类实例自身；
    # x_enc 是编码器的输入数据，即时间序列本身；x_mark_enc 是编码器输入数据的时间标记；
    # x_dec 是解码器的输入数据；x_mark_dec 是解码器输入数据的时间标记；
    # mask 是可选参数，用于处理某些任务中的掩码（比如在缺失值填补中可能需要用到）。
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 如果任务是预测类型（长期或短期），那么调用模型的 forecast 方法，
            # 传入编码器的输入数据 x_enc、时间标记 x_mark_enc、解码器输入数据 x_dec 和时间标记 x_mark_dec。
            # forecast 方法会根据输入数据生成预测的输出。这里的 dec_out 代表解码器的输出，即预测结果。
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            # 返回 dec_out 的一部分。
            # dec_out[:, -self.pred_len:, :] 表示获取 dec_out 中时间维度上最后 self.pred_len 步的输出，即预测部分的数据。
            # [B, L, D] 是返回张量的形状说明：B 表示批次大小（batch size），L 表示预测的长度 self.pred_len，D 表示特征的数量。
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            # 如果任务是缺失值填补，那么调用模型的 imputation 方法，并传入编码器的输入数据 x_enc、时间标记 x_mark_enc、
            # 解码器输入数据 x_dec、时间标记 x_mark_dec 和掩码 mask。
            # 该方法将根据输入和掩码生成填补后的时间序列。dec_out 是解码器的输出，即补全后的时间序列。
            return dec_out  # [B, L, D]
            # 返回 dec_out，它的形状为 [B, L, D]，表示经过填补的时间序列。B 是批次大小，L 是时间长度，D 是特征数量。
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            # 如果任务是异常检测，那么调用模型的 anomaly_detection 方法，并传入编码器的输入数据 x_enc。
            # 该方法将根据输入时间序列检测异常点。dec_out 是解码器的输出，包含异常检测的结果。
            return dec_out  # [B, L, D]
            # 返回 dec_out，它的形状为 [B, L, D]，表示异常检测后的时间序列结果。B 是批次大小，L 是时间长度，D 是特征数量。
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            # 如果任务是分类，那么调用模型的 classification 方法，并传入编码器的输入数据 x_enc 和时间标记 x_mark_enc。
            # 该方法根据输入数据生成分类结果。dec_out 是解码器的输出，即分类的结果。
            return dec_out  # [B, N]
            # 返回 dec_out，其形状为 [B, N]，表示分类的输出结果。B 是批次大小，N 是类别数量，即每个样本所属的类别。
        return None
        # 如果没有匹配的任务类型（即 task_name 不属于预测、缺失值填补、异常检测或分类），则返回 None。
