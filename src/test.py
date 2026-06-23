import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data.datasets.epf import EPF, EPFInfo
from utils.pytorch.ts_dataset import TimeSeriesDataset
from utils.pytorch.ts_loader import TimeSeriesLoader
from nbeats.nbeats import Nbeats


def save_model_and_params(model, save_dir='saved_models'):
    """
    保存模型和参数
    """
    import os
    import json
    import pickle

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 直接保存整个模型对象
    model_path = os.path.join(save_dir, 'nbeats_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # 使用实际的训练配置参数
    model_config = {
        'input_size_multiplier': 12,
        'output_size': 1,
        'shared_weights': False,
        'initialization': 'glorot_normal',
        'activation': 'relu',
        'stack_types': ['trend', 'seasonality', 'exogenous_tcn'],
        'n_blocks': [2, 2, 2],
        'n_layers': [1, 1, 1],
        'n_hidden': [[200, 200], [200, 200], [200, 200]],
        'n_harmonics': 1,
        'n_polynomials': 100,
        'x_s_n_hidden': 0,
        'exogenous_n_channels': 2,
        'include_var_dict': model.include_var_dict,  # 从模型获取
        't_cols': model.t_cols,  # 从模型获取
        'batch_normalization': True,
        'dropout_prob_theta': 0.2,
        'dropout_prob_exogenous': 0,
        'learning_rate': 0.001,
        'lr_decay': 0.9,
        'n_lr_decay_steps': 10,
        'early_stopping': 300,
        'weight_decay': 0,
        'l1_theta': 0,
        'n_iterations': 500,
        'loss': 'MAE',
        'loss_hypar': 1,
        'val_loss': 'MAE',
        'seasonality': 1,
        'random_seed': 32,
        'input_size': len(model.include_var_dict['y'])  # 从include_var_dict获取input_size
    }

    # 保存配置参数
    params_path = os.path.join(save_dir, 'model_params.json')
    with open(params_path, 'w') as f:
        json.dump(model_config, f, indent=4)

    print(f"\n模型和参数已保存到 {save_dir}")
    print("\n关键参数验证:")
    key_params = ['input_size', 'output_size', 'n_blocks', 'n_hidden',
                 'learning_rate', 'n_iterations', 'n_harmonics', 'n_polynomials']
    for param in key_params:
        print(f"{param}: {model_config[param]}")

    return model_config

# Load your data from a CSV file
data = pd.read_csv(r"D:\TempFile\WDS\PycharmProject\Predict_code\nbeatsx-main\src\桡足类与温度2022.csv")
#data['ImageName'] = pd.to_datetime(data['ImageName'].astype(str).str.strip(), format='%Y%m%d%H%M%S%f', errors='coerce')
data['Hour'] = pd.to_datetime(data['Hour'])
#data['Hour'] = pd.to_datetime(data['Hour'], format="%Y/%m/%d %H:%M:%S")
Y_df = data[['Hour', 'Copepoda']].rename(columns={'Copepoda': 'y', 'Hour': 'ds'})
Y_df['unique_id'] = 'Copepoda'  # 所有 unique_id 列均为 'Noctiluca'
# 将 unique_id 列放到第一列
Y_df = Y_df[['unique_id', 'ds', 'y']]
print(Y_df.head())
timestamps = data['Hour'].values

X_df = pd.DataFrame()

# 添加 unique_id 列和 ds 列
X_df['unique_id'] = ['Copepoda'] * len(timestamps)  # 确保 unique_id 列正确填充
X_df['ds'] = timestamps            #  添加时间戳列
X_df['Temp_C'] = data['Temp_C']
print('@@@103', data['Sal'])
X_df['Sal'] = data['Sal']-data['Sal'].mean()

# print('@@@@@98',data['Temp_C'],data['Sal'],data['Sal'].mean())
'''
# 保留源数据中排除 'ImageName' 和 'Noctiluca' 的列
for column in data.columns[1:]:  # 从第二列开始遍历
    if column != 'Noctiluca':  # 排除 Noctiluca 列
        X_df[column] = data[column].values  # 确保使用 .values 填充数据
'''
t
# 打印结果
print(X_df.head())  # 打印结果

print("Y_df 行数:", len(Y_df))
#print("X_df 行数:", len(X_df))
# 设置训练掩码：前1800行作为训练数据
train_mask = np.ones(len(Y_df))
train_mask[-12:] = 0  # 使用后面的数据作为测试集
print(f"train_mask length: {len(train_mask)}, expected max_len: {len(Y_df)}")
train_mask_1 = train_mask.tolist()  # 将 ndarray 转换为 list
# 创建数据集对象，预处理 DataFrame 为 PyTorch 张量和窗口
ts_dataset = TimeSeriesDataset(Y_df=Y_df,X_df=X_df,ts_train_mask = train_mask_1)
print(ts_dataset.t_cols)  # 打印结果
# 创建训练和验证加载器
train_loader = TimeSeriesLoader(model='nbeats',
                                ts_dataset=ts_dataset,
                                window_sampling_limit=24,
                                offset=0,
                                input_size=12,
                                output_size=1,
                                idx_to_sample_freq=1,
                                batch_size=8,
                                is_train_loader=True,
                                shuffle=False,)

val_loader = TimeSeriesLoader(model='nbeats',
                              ts_dataset=ts_dataset,
                              window_sampling_limit=24,
                              offset=0,
                              input_size=12,
                              output_size=1,
                              idx_to_sample_freq=1,
                              batch_size=8,
                              is_train_loader=False,
                              shuffle=False)
# # 包含滞后的字典
include_var_dict = {'y': [0,1],'Sal':[0,1],'Temp_C':[0,1]}
# 确保外生变量的名字与数据中的列名匹配
#
# 初始化模型
model = Nbeats(input_size_multiplier=12,
               output_size=1,
               shared_weights=False,
               initialization='glorot_normal',
               activation='relu',
               stack_types=[ 'trend', 'seasonality','exogenous_tcn'],
               n_blocks=[2 ,2, 2],   #增加每个stack中的块数为3
               n_layers=[1, 1, 1],
               n_hidden=[[200, 200], [200, 200],  [200, 200]],
               n_harmonics=1,
               n_polynomials=0,
               x_s_n_hidden=0,
               exogenous_n_channels=2,
               include_var_dict=include_var_dict,
               t_cols=ts_dataset.t_cols,
               batch_normalization=True,
               dropout_prob_theta=0.2,
               dropout_prob_exogenous=0,
               learning_rate=0.001,
               lr_decay=0.9,
               n_lr_decay_steps=10,
               early_stopping=250,
               weight_decay=0,
               l1_theta=0,
               n_iterations=100,
               loss='MAE',
               loss_hypar=1,
               val_loss='MAE',
               seasonality=1,
               random_seed=32)

# 训练模型
model.fit(train_ts_loader=train_loader, val_ts_loader=val_loader, eval_steps=12)
save_model_and_params(model, save_dir='saved_models')
y_true, y_hat,  block_forecasts,cumulative_forecasts, *_ = model.predict(ts_loader=val_loader, return_decomposition=True)
y_hat_df = pd.DataFrame(y_hat.flatten())
y_hat_df.to_csv(r'D:\TempFile\WDS\PycharmProject\Predict_code\nbeatsx-main\src\pred_cop_2022全年预测.csv', index=False)
y_hat = y_hat.flatten()[-12:]
y_hat[y_hat < 0] = 0
plt.plot(range(12), Y_df['y'].values[-12:], label='number')
plt.plot(range(12), y_hat.flatten(), linestyle='dashed', label='Forecast')
#plt.axvline(700, color='black')
plt.legend()
plt.grid()
plt.xlabel('HOUR')
plt.ylabel('Copepoda')
plt.show()

print("block_forecasts shape:", block_forecasts.shape)
trend_pred= block_forecasts[:, 0, :].flatten()        # 提取 trend
seasonality_pred= block_forecasts[:, 1, :].flatten()  # 提取 seasonality
exogenous_pred = block_forecasts[:, 2, :].flatten()    # 提取 exogenous

# 只选测试集部分数据
trend_pred = trend_pred[-12:]
seasonality_pred = seasonality_pred[-12:]
exogenous_pred = exogenous_pred[-12:]

# 绘制模块输出
plt.figure(figsize=(12, 8))
plt.plot(range(12), trend_pred, label='Trend Component', color='red')
plt.plot(range(12), seasonality_pred, label='Seasonality Component', color='green')
plt.plot(range(12), exogenous_pred, label='salinity Component', color='orange')

#plt.axvline(700, color='black', linestyle='dotted')
plt.legend()
plt.grid()
plt.xlabel('HOUR')
plt.ylabel('calibration deviation')
plt.title('1-7 month 2022 year Copepoda Decomposition of Forecast')
plt.show()


