import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data.datasets.epf import EPF, EPFInfo
from utils.pytorch.ts_dataset import TimeSeriesDataset
from utils.pytorch.ts_loader import TimeSeriesLoader
from nbeats.nbeats import Nbeats


def complete_training_timeline(raw_data, freq):
    """
    按时间频率补齐训练数据。
    freq='H' 用于小时模型，freq='D' 用于日模型。
    缺失时间点的数值字段使用前后时间点插值；单个缺失点等价于取前后均值。
    边界无法插值时再补 0。
    """
    if raw_data.empty:
        return raw_data

    data = raw_data.copy()
    data['SnapTime'] = pd.to_datetime(data['SnapTime'], errors='coerce')
    data = data.dropna(subset=['SnapTime'])
    data = data.sort_values('SnapTime').drop_duplicates(subset=['SnapTime'], keep='last')

    full_time_index = pd.date_range(data['SnapTime'].min(), data['SnapTime'].max(), freq=freq)
    data = data.set_index('SnapTime').reindex(full_time_index)
    data.index.name = 'SnapTime'

    for column_name in data.columns:
        data[column_name] = pd.to_numeric(data[column_name], errors='coerce')
    data = data.interpolate(method='linear', limit_direction='both').fillna(0.0)
    data = data.reset_index()
    return data


def save_model_and_params(model,train_model_bio_i, save_dir='saved_models', model_file_key=None,
                          input_size_multiplier=12, output_size=1, n_iterations=500):
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
    if model_file_key is None:
        model_file_key = train_model_bio_i

    model_path = os.path.join(save_dir, model_file_key+'_nbeats_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # 使用实际的训练配置参数
    model_config = {
        'input_size_multiplier': input_size_multiplier,
        'output_size': output_size,
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
        'n_iterations': n_iterations,
        'loss': 'MAE',
        'loss_hypar': 1,
        'val_loss': 'MAE',
        'seasonality': 1,
        'random_seed': 32,
        'input_size': len(model.include_var_dict['y'])  # 从include_var_dict获取input_size
    }

    # 保存配置参数
    params_path = os.path.join(save_dir, model_file_key+'_model_params.json')
    with open(params_path, 'w') as f:
        json.dump(model_config, f, indent=4)

    print(f"\n模型和参数已保存到 {save_dir}")
    print("\n关键参数验证:")
    key_params = ['input_size', 'output_size', 'n_blocks', 'n_hidden',
                 'learning_rate', 'n_iterations', 'n_harmonics', 'n_polynomials']
    for param in key_params:
        print(f"{param}: {model_config[param]}")

    return model_config

def read_12h_train_tab_to_train_model_fun(datebase_conn,tab_train_12h_name,need_predict_bio_arr):
    # 读取整个表
    query = f"SELECT * FROM {tab_train_12h_name}"
    h12_all_train_data = pd.read_sql(query, datebase_conn)
    h12_all_train_data = complete_training_timeline(h12_all_train_data, freq='H')
    data_row = h12_all_train_data.shape[0]
    data_col = h12_all_train_data.shape[1]
    ####如果表中数据小于24行则不训练
    if data_row < 24:
        print(f"12h训练数据不足，当前 {data_row} 行，需要至少 24 行")
        return False

    print('h12_all_train_data',h12_all_train_data,h12_all_train_data.shape[0],h12_all_train_data.shape[1])
    for train_model_bio_i in need_predict_bio_arr:
        data = h12_all_train_data.copy()
        Y_df = data[['SnapTime', train_model_bio_i]].rename(columns={train_model_bio_i: 'y', 'SnapTime': 'ds'})
        Y_df['unique_id'] = train_model_bio_i  # 所有 unique_id 列均为 'Noctiluca'
        # 将 unique_id 列放到第一列
        Y_df = Y_df[['unique_id', 'ds', 'y']]
        print(Y_df.head())
        timestamps = data['SnapTime'].values

        X_df = pd.DataFrame()

        # 添加 unique_id 列和 ds 列
        X_df['unique_id'] = [train_model_bio_i] * len(timestamps)  # 确保 unique_id 列正确填充
        X_df['ds'] = timestamps  # 添加时间戳列
        X_df['Temperature'] = data['Temperature']
        print('@@@103',type(data['Salinity']), data['Salinity'])
        X_df['Salinity'] = data['Salinity'] - data['Salinity'].mean()


        # 打印结果
        print(X_df.head())  # 打印结果

        print("Y_df 行数:", len(Y_df))
        # print("X_df 行数:", len(X_df))
        # 设置训练掩码：前1800行作为训练数据
        train_mask = np.ones(len(Y_df))
        train_mask[-12:] = 0  # 使用后面的数据作为测试集
        print(f"train_mask length: {len(train_mask)}, expected max_len: {len(Y_df)}")
        train_mask_1 = train_mask.tolist()  # 将 ndarray 转换为 list
        # 创建数据集对象，预处理 DataFrame 为 PyTorch 张量和窗口
        ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, ts_train_mask=train_mask_1)
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
                                        shuffle=False, )

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
        include_var_dict = {'y': [0, 1], 'Salinity': [0, 1], 'Temperature': [0, 1]}
        # 确保外生变量的名字与数据中的列名匹配
        #
        # 初始化模型
        model = Nbeats(input_size_multiplier=12,
                       output_size=1,
                       shared_weights=False,
                       initialization='glorot_normal',
                       activation='relu',
                       stack_types=['trend', 'seasonality', 'exogenous_tcn'],
                       n_blocks=[2, 2, 2],  # 增加每个stack中的块数为3
                       n_layers=[1, 1, 1],
                       n_hidden=[[200, 200], [200, 200], [200, 200]],
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
        save_model_and_params(model, train_model_bio_i,save_dir='saved_models')
        print('训练完成！')
    return True


def read_7d_train_tab_to_train_model_fun(datebase_conn, tab_train_7d_name, need_predict_bio_arr):
    """
    从日级训练表训练 7d -> 1d 模型。
    日级训练表字段沿用 SnapTime、物种字段、Temperature、Salinity。
    """
    query = f"SELECT * FROM {tab_train_7d_name}"
    d7_all_train_data = pd.read_sql(query, datebase_conn)
    d7_all_train_data = complete_training_timeline(d7_all_train_data, freq='D')
    data_row = d7_all_train_data.shape[0]
    if data_row <= 14:
        print(f"7d训练数据不足，当前 {data_row} 行，需要大于 14 行")
        return False

    print('d7_all_train_data', d7_all_train_data, d7_all_train_data.shape[0], d7_all_train_data.shape[1])
    for train_model_bio_i in need_predict_bio_arr:
        data = d7_all_train_data.copy()

        Y_df = data[['SnapTime', train_model_bio_i]].rename(columns={train_model_bio_i: 'y', 'SnapTime': 'ds'})
        Y_df['unique_id'] = train_model_bio_i
        Y_df = Y_df[['unique_id', 'ds', 'y']]

        timestamps = data['SnapTime'].values
        X_df = pd.DataFrame()
        X_df['unique_id'] = [train_model_bio_i] * len(timestamps)
        X_df['ds'] = timestamps
        X_df['Temperature'] = pd.to_numeric(data['Temperature'], errors='coerce').fillna(0.0)
        salinity = pd.to_numeric(data['Salinity'], errors='coerce').fillna(0.0)
        X_df['Salinity'] = salinity - salinity.mean()

        train_mask = np.ones(len(Y_df))
        train_mask[-7:] = 0
        train_mask_1 = train_mask.tolist()

        ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, ts_train_mask=train_mask_1)
        train_loader = TimeSeriesLoader(model='nbeats',
                                        ts_dataset=ts_dataset,
                                        window_sampling_limit=14,
                                        offset=0,
                                        input_size=7,
                                        output_size=1,
                                        idx_to_sample_freq=1,
                                        batch_size=8,
                                        is_train_loader=True,
                                        shuffle=False)

        val_loader = TimeSeriesLoader(model='nbeats',
                                      ts_dataset=ts_dataset,
                                      window_sampling_limit=14,
                                      offset=0,
                                      input_size=7,
                                      output_size=1,
                                      idx_to_sample_freq=1,
                                      batch_size=8,
                                      is_train_loader=False,
                                      shuffle=False)

        include_var_dict = {'y': [0, 1], 'Salinity': [0, 1], 'Temperature': [0, 1]}
        model = Nbeats(input_size_multiplier=7,
                       output_size=1,
                       shared_weights=False,
                       initialization='glorot_normal',
                       activation='relu',
                       stack_types=['trend', 'seasonality', 'exogenous_tcn'],
                       n_blocks=[2, 2, 2],
                       n_layers=[1, 1, 1],
                       n_hidden=[[200, 200], [200, 200], [200, 200]],
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

        model.fit(train_ts_loader=train_loader, val_ts_loader=val_loader, eval_steps=7)
        save_model_and_params(model,
                              train_model_bio_i,
                              save_dir='saved_models',
                              model_file_key=train_model_bio_i+'_d7',
                              input_size_multiplier=7,
                              output_size=1,
                              n_iterations=100)
        print(f'{train_model_bio_i} 7d模型训练完成！')
    return True







import pymysql
from all_function import mysqlConnect
if __name__ == '__main__':
    data_ip_set = 'localhost'
    data_port_set = 3306
    data_user_set = 'root'
    data_password_set = 'root'
    data_name_set = 'szsw'

    test_tab_root_name = 'tab_avg_root'
    tab_train_12h_name = 'tab_train_12h'
    tab_train_7d_name = 'tab_train_7d'
    tab_predict_12h_temp_name = 'tab_predict_12h_temp'
    tab_predict_7d_temp_name = 'tab_predict_7d_temp'
    tab_predict_12h_record_name = 'tab_predict_12h_record'
    tab_predict_7d_record_name = 'tab_predict_7d_record'
    datebase_conn = mysqlConnect(data_ip_set, data_port_set, data_user_set, data_password_set, data_name_set)
    need_predict_bio_arr = ['Copepoda','Medusae']
    read_12h_train_tab_to_train_model_fun(datebase_conn,tab_train_12h_name,need_predict_bio_arr)

