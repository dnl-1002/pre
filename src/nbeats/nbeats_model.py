import math
import torch as t
import torch.nn as nn
from typing import Tuple
from nbeats.tcn import TemporalConvNet
import pandas as pd
import numpy as np
import pickle
import os
from utils.pytorch.ts_dataset import TimeSeriesDataset
from utils.pytorch.ts_loader import TimeSeriesLoader

####过去7d预测未来1d的值
def d7_predict_next_1d_points(historical_data,bio_i_str, model_dir='saved_models'):
    """
    预测接口函数

    参数:
    historical_data: 字典，包含过去12个点的数据，格式如下：
        {
            'y': [v1, v2, ..., v12],        # 目标变量(桡足类)最近12个点的值
            'Temp_C': [t1, t2, ..., t12],   # 温度最近12个点的值
            'Sal': [s1, s2, ..., s12]       # 盐度最近12个点的值
        }
    model_dir: 保存模型的目录路径

    返回:
    predictions: numpy数组，包含未来2个时间点的预测值
    """
    Salinity_str = 'Salinity'
    Temperature_str ='Temperature'
    try:
        # 1. 数据验证
        required_length = 7  # 需要的历史数据长度
        for key, values in historical_data.items():
            if len(values) != required_length:
                raise ValueError(f"{key} 需要 {required_length} 个历史数据点，但提供了 {len(values)} 个")

        # 2. 加载模型
        model_path = os.path.join(model_dir, 'nbeats_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        sal_mean = np.mean(historical_data[Salinity_str])
        sal_centered = [x - sal_mean for x in historical_data[Salinity_str]]

        # 3. 准备数据
        # 创建时间索引
        current_time = pd.Timestamp.now()
        time_index = pd.date_range(end=current_time, periods=required_length, freq='D')

        # 创建Y_df
        Y_df = pd.DataFrame({
            'unique_id': [bio_i_str] * required_length,
            'ds': time_index,
            'y': historical_data['y']
        })

        # 创建X_df
        X_df = pd.DataFrame({
            'unique_id': [bio_i_str] * required_length,
            'ds': time_index,
            Temperature_str: historical_data[Temperature_str],
            Salinity_str: sal_centered
        })

        # 4. 创建数据集和加载器
        ts_dataset = TimeSeriesDataset(
            Y_df=Y_df,
            X_df=X_df,
            ts_train_mask=[0] * required_length  # 测试模式
        )

        test_loader = TimeSeriesLoader(
            model='nbeats',
            ts_dataset=ts_dataset,
            window_sampling_limit=12,  # 使用与训练时相同的值
            offset=0,
            input_size=12,
            output_size=1,
            idx_to_sample_freq=1,
            batch_size=8,
            is_train_loader=False,
            shuffle=False
        )

        # 5. 进行预测
        _, y_hat, *_ = model.predict(ts_loader=test_loader)
        predictions = y_hat.flatten()

        return predictions

    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")
        raise






####过去12h预测未来1h的值
def h12_predict_next_1h_points(historical_data,bio_i_str, model_dir='saved_models'):
    """
    预测接口函数

    参数:
    historical_data: 字典，包含过去12个点的数据，格式如下：
        {
            'y': [v1, v2, ..., v12],        # 目标变量(桡足类)最近12个点的值
            'Temp_C': [t1, t2, ..., t12],   # 温度最近12个点的值
            'Sal': [s1, s2, ..., s12]       # 盐度最近12个点的值
        }
    model_dir: 保存模型的目录路径

    返回:
    predictions: numpy数组，包含未来2个时间点的预测值
    """
    Salinity_str = 'Salinity'
    Temperature_str ='Temperature'
    try:
        # 1. 数据验证
        required_length = 12  # 需要的历史数据长度
        for key, values in historical_data.items():
            if len(values) != required_length:
                raise ValueError(f"{key} 需要 {required_length} 个历史数据点，但提供了 {len(values)} 个")

        # 2. 加载模型
        model_path = os.path.join(model_dir, bio_i_str+'_nbeats_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        sal_mean = np.mean(historical_data[Salinity_str])
        sal_centered = [x - sal_mean for x in historical_data[Salinity_str]]

        # 3. 准备数据
        # 创建时间索引
        current_time = pd.Timestamp.now()
        time_index = pd.date_range(end=current_time, periods=required_length, freq='H')

        # 创建Y_df
        Y_df = pd.DataFrame({
            'unique_id': [bio_i_str] * required_length,
            'ds': time_index,
            'y': historical_data['y']
        })

        # 创建X_df
        X_df = pd.DataFrame({
            'unique_id': [bio_i_str] * required_length,
            'ds': time_index,
            Temperature_str: historical_data[Temperature_str],
            Salinity_str: sal_centered
        })

        # 4. 创建数据集和加载器
        ts_dataset = TimeSeriesDataset(
            Y_df=Y_df,
            X_df=X_df,
            ts_train_mask=[0] * required_length  # 测试模式
        )

        test_loader = TimeSeriesLoader(
            model='nbeats',
            ts_dataset=ts_dataset,
            window_sampling_limit=12,  # 使用与训练时相同的值
            offset=0,
            input_size=12,
            output_size=1,
            idx_to_sample_freq=1,
            batch_size=8,
            is_train_loader=False,
            shuffle=False
        )

        # 5. 进行预测
        _, y_hat, *_ = model.predict(ts_loader=test_loader)
        predictions = y_hat.flatten()

        return predictions

    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")
        raise


def filter_input_vars(insample_y, insample_x_t, outsample_x_t, t_cols, include_var_dict):
    # This function is specific for the EPF task
    if t.cuda.is_available():
        device = insample_x_t.get_device()
    else:
        device = 'cpu'
    outsample_y = t.zeros((insample_y.shape[0], 1, outsample_x_t.shape[2])).to(device)

    insample_y_aux = t.unsqueeze(insample_y,dim=1)

    insample_x_t_aux = t.cat([insample_y_aux, insample_x_t], dim=1)
    outsample_x_t_aux = t.cat([outsample_y, outsample_x_t], dim=1)
    x_t = t.cat([insample_x_t_aux, outsample_x_t_aux], dim=-1)
    batch_size, n_channels, input_size = x_t.shape

    assert input_size, f'input_size {input_size} '

    x_t = x_t.reshape(batch_size, n_channels, input_size, 1)

    input_vars = []
    for var in include_var_dict.keys():
        if len(include_var_dict[var])>0:
            t_col_idx    = t_cols.index(var)
            t_col_filter = include_var_dict[var]
            if var != 'week_day':
                input_vars  += [x_t[:, t_col_idx, t_col_filter, :]]
            else:
                assert t_col_filter == [-1], f'Day of week must be of outsample not {t_col_filter}'
                day_var = x_t[:, t_col_idx, t_col_filter, [0]]
                day_var = day_var.view(batch_size, -1)

    x_t_filter = t.cat(input_vars, dim=1)
    x_t_filter = x_t_filter.view(batch_size,-1)

#    if len(include_var_dict['week_day'])>0:
#       x_t_filter = t.cat([x_t_filter, day_var], dim=1)

    return x_t_filter

class _StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(_StaticFeaturesEncoder, self).__init__()
        layers = [nn.Dropout(p=0.5),
                  nn.Linear(in_features=in_features, out_features=out_features),
                  nn.ReLU()]
        self.encoder = nn.Sequential(*layers)


    def forward(self, x):
        x = self.encoder(x)
        return x

class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self, x_t_n_inputs: int, x_s_n_inputs: int, x_s_n_hidden: int, theta_n_dim: int, basis: nn.Module,
                 n_layers: int, theta_n_hidden: list, include_var_dict, t_cols, batch_normalization: bool, dropout_prob: float, activation: str):
        """
        """
        super().__init__()

        if x_s_n_inputs == 0:
            x_s_n_hidden = 0
        theta_n_hidden = [x_t_n_inputs + x_s_n_hidden] + theta_n_hidden#[200,200,200,200,200]

        self.x_s_n_inputs = x_s_n_inputs
        self.x_s_n_hidden = x_s_n_hidden
        self.include_var_dict = include_var_dict
        self.t_cols = t_cols
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        self.activations = {'relu': nn.ReLU(),
                            'softplus': nn.Softplus(),
                            'tanh': nn.Tanh(),
                            'selu': nn.SELU(),
                            'lrelu': nn.LeakyReLU(),
                            'prelu': nn.PReLU(),
                            'sigmoid': nn.Sigmoid()}

        hidden_layers = []
        for i in range(n_layers):

            # Batch norm after activation
            hidden_layers.append(nn.Linear(in_features=theta_n_hidden[i], out_features=theta_n_hidden[i+1]))
            hidden_layers.append(self.activations[activation])

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=theta_n_hidden[i+1]))

            if self.dropout_prob>0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=theta_n_hidden[-1], out_features=theta_n_dim)]
        layers = hidden_layers + output_layer

        # x_s_n_inputs is computed with data, x_s_n_hidden is provided by user, if 0 no statics are used
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            self.static_encoder = _StaticFeaturesEncoder(in_features=x_s_n_inputs, out_features=x_s_n_hidden)
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor,
                outsample_x_t: t.Tensor, x_s: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:

        if self.include_var_dict is not None:
            insample_y = filter_input_vars(insample_y=insample_y, insample_x_t=insample_x_t, outsample_x_t=outsample_x_t,
                                           t_cols=self.t_cols, include_var_dict=self.include_var_dict)

        # Static exogenous
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            x_s = self.static_encoder(x_s)
            insample_y = t.cat((insample_y, x_s), 1)
        #print(insample_y.shape)
        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)

        return backcast, forecast

class NBeats(nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
                outsample_x_t: t.Tensor, x_s: t.Tensor, return_decomposition=False):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast =0   # Level with Naive1 insample_y[:,-1:]
        block_forecasts = []
        cumulative_forecasts = []  # 用于存储每个模块的独立预测值

        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals, insample_x_t=insample_x_t,
                                             outsample_x_t=outsample_x_t, x_s=x_s)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)
            cumulative_forecasts.append(forecast.clone())  # 存储当前模块预测的累计结果

        # (n_batch, n_blocks, n_time)
        block_forecasts = t.stack(block_forecasts).permute(1, 0, 2)
        cumulative_forecasts = t.stack(cumulative_forecasts).permute(1, 0, 2)

        if return_decomposition:
            return forecast, block_forecasts, cumulative_forecasts
        else:
            return forecast
    '''
    def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
                outsample_x_t: t.Tensor, x_s: t.Tensor, return_decomposition = False):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:] # Level with Naive1
        block_forecasts = []
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals, insample_x_t=insample_x_t,
                                             outsample_x_t=outsample_x_t, x_s=x_s)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        # (n_batch, n_blocks, n_time)

        block_forecasts = t.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1,0,2)
        #print("Inside forward, block_forecasts shape:", block_forecasts.shape)

        if return_decomposition:
            return forecast, block_forecasts
        else:
            return forecast'''

    def decomposed_prediction(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
                              outsample_x_t: t.Tensor):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:] # Level with Naive1
        forecast_components = []
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals, insample_x_t, outsample_x_t)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            forecast_components.append(block_forecast)
        return forecast, forecast_components

class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, -self.forecast_size:]
        return backcast, forecast

class TrendBasis(nn.Module):
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        polynomial_size = degree_of_polynomial + 1
        self.backcast_basis = nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=float) / backcast_size, i)[None, :]
                                    for i in range(polynomial_size)]), dtype=t.float32), requires_grad=False)
        self.forecast_basis = nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=float) / forecast_size, i)[None, :]
                                    for i in range(polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = t.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast

class SeasonalityBasis(nn.Module):
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        frequency = np.append(np.zeros(1, dtype=np.float32),
                                        np.arange(harmonics, harmonics / 2 * forecast_size,
                                                    dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * frequency

        backcast_cos_template = t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32)
        backcast_sin_template = t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32)
        backcast_template = t.cat([backcast_cos_template, backcast_sin_template], dim=0)

        forecast_cos_template = t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32)
        forecast_sin_template = t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32)
        forecast_template = t.cat([forecast_cos_template, forecast_sin_template], dim=0)

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = t.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast

class ExogenousBasisInterpretable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis = insample_x_t
        forecast_basis = outsample_x_t

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ExogenousBasisWavenet(nn.Module):
    def __init__(self, out_features, in_features, num_levels=4, kernel_size=3, dropout_prob=0):
        super().__init__()
        # Shape of (1, in_features, 1) to broadcast over b and t
        self.weight = nn.Parameter(t.Tensor(1, in_features, 1), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(0.5))

        padding = (kernel_size - 1) * (2**0)
        input_layer = [nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                 kernel_size=kernel_size, padding=padding, dilation=2**0),
                                 Chomp1d(padding),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_prob)]
        conv_layers = []
        for i in range(1, num_levels):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            conv_layers.append(nn.Conv1d(in_channels=out_features, out_channels=out_features,
                                         padding=padding, kernel_size=3, dilation=dilation))
            conv_layers.append(Chomp1d(padding))
            conv_layers.append(nn.ReLU())
        conv_layers = input_layer + conv_layers

        self.wavenet = nn.Sequential(*conv_layers)

    def transform(self, insample_x_t, outsample_x_t):
        input_size = insample_x_t.shape[2]

        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)

        x_t = x_t * self.weight # Element-wise multiplication, broadcasted on b and t. Weights used in L1 regularization
        x_t = self.wavenet(x_t)[:]

        backcast_basis = x_t[:,:, :input_size]
        forecast_basis = x_t[:,:, input_size:]

        return backcast_basis, forecast_basis

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast

class ExogenousBasisTCN(nn.Module):
    def __init__(self, out_features, in_features, num_levels = 4, kernel_size=2, dropout_prob=0.1):#TCN模型调参位置
        super().__init__()
        n_channels = num_levels * [out_features]
        self.tcn = TemporalConvNet(num_inputs=in_features, num_channels=n_channels, kernel_size=kernel_size, dropout=dropout_prob)

    def transform(self, insample_x_t, outsample_x_t):
        input_size = insample_x_t.shape[2]

        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)

        x_t = self.tcn(x_t)[:]
        backcast_basis = x_t[:,:, :input_size]
        forecast_basis = x_t[:,:, input_size:]

        return backcast_basis, forecast_basis

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast