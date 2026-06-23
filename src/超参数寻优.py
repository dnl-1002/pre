import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import numpy as np
from itertools import product
import logging
import traceback
from utils.data.datasets.epf import EPF, EPFInfo
from utils.pytorch.ts_dataset import TimeSeriesDataset
from utils.pytorch.ts_loader import TimeSeriesLoader
from nbeats.nbeats import Nbeats

from sklearn.preprocessing import LabelEncoder

# Load your data from a CSV file
data = pd.read_csv("E:/plankton_pre/桡足类与温度2022.csv")
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
X_df['Sal'] = data['Sal']-data['Sal'].mean()
'''
# 保留源数据中排除 'ImageName' 和 'Noctiluca' 的列
for column in data.columns[1:]:  # 从第二列开始遍历
    if column != 'Noctiluca':  # 排除 Noctiluca 列
        X_df[column] = data[column].values  # 确保使用 .values 填充数据
'''
# 打印结果
print(X_df.head())  # 打印结果

print("Y_df 行数:", len(Y_df))
#print("X_df 行数:", len(X_df))
# 设置训练掩码：前1800行作为训练数据
train_mask = np.ones(len(Y_df))
train_mask[-1500:] = 0  # 使用后面的数据作为测试集
print(f"train_mask length: {len(train_mask)}, expected max_len: {len(Y_df)}")
train_mask_1 = train_mask.tolist()  # 将 ndarray 转换为 list
# 创建数据集对象，预处理 DataFrame 为 PyTorch 张量和窗口
ts_dataset = TimeSeriesDataset(Y_df=Y_df,X_df=X_df,ts_train_mask = train_mask_1)
print(ts_dataset.t_cols)  # 打印结果


class ParameterSensitivityAnalysis:
    def __init__(self, tuning_results):
        """
        初始化参数敏感性分析

        参数:
        tuning_results: 超参数调优的结果列表，每个元素包含 'params' 和 'score'
        """
        self.results_df = self._prepare_results_df(tuning_results)

    def _prepare_results_df(self, tuning_results):
        """
        将调优结果转换为DataFrame格式
        """
        # 展平参数字典
        flat_results = []
        for result in tuning_results:
            flat_dict = self._flatten_dict(result['params'])
            flat_dict['score'] = result['score']
            flat_results.append(flat_dict)

        return pd.DataFrame(flat_results)

    def _flatten_dict(self, d, parent_key='', sep='_'):
        """
        展平嵌套字典，处理列表和嵌套结构
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, (list, tuple)):
                if any(isinstance(x, (list, tuple)) for x in v):
                    # 处理嵌套列表，转换为字符串
                    items.append((new_key, str(v)))
                else:
                    # 处理简单列表
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def plot_parameter_distributions(self, figsize=(15, 10)):
        """
        绘制不同参数值对应的模型性能分布
        """
        numeric_cols = self.results_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'score']

        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.scatterplot(data=self.results_df, x=col, y='score', ax=axes[i])
            axes[i].set_title(f'{col} vs Score')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Score (MAE)')

        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig

    def calculate_parameter_importance(self):
        """
        计算参数重要性（基于相关性分析）
        """
        numeric_cols = self.results_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'score']

        importance_dict = {}
        for col in numeric_cols:
            correlation, p_value = spearmanr(self.results_df[col], self.results_df['score'])
            importance_dict[col] = {
                'correlation': abs(correlation),
                'p_value': p_value
            }

        return pd.DataFrame(importance_dict).T.sort_values('correlation', ascending=False)

    def plot_parameter_importance(self, figsize=(10, 6)):
        """
        可视化参数重要性
        """
        importance_df = self.calculate_parameter_importance()

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x=importance_df.index, y='correlation', data=importance_df)
        plt.xticks(rotation=45, ha='right')
        plt.title('Parameter Importance (Based on Correlation with Score)')
        plt.tight_layout()
        return fig

    def analyze_interactions(self, top_n=3):
        """
        分析参数之间的交互作用
        """
        numeric_cols = self.results_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'score']

        # 获取最重要的参数
        importance_df = self.calculate_parameter_importance()
        top_params = importance_df.head(top_n).index.tolist()

        # 创建交互图
        fig = plt.figure(figsize=(15, 5 * (top_n - 1)))
        plot_idx = 1

        for i in range(len(top_params)):
            for j in range(i + 1, len(top_params)):
                plt.subplot(top_n - 1, top_n - 1, plot_idx)
                param1, param2 = top_params[i], top_params[j]

                scatter = plt.scatter(
                    self.results_df[param1],
                    self.results_df[param2],
                    c=self.results_df['score'],
                    cmap='viridis'
                )
                plt.colorbar(scatter, label='Score')
                plt.xlabel(param1)
                plt.ylabel(param2)
                plt.title(f'Interaction: {param1} vs {param2}')
                plot_idx += 1

        plt.tight_layout()
        return fig


def analyze_sensitivity(tuning_results):
    """
    主函数：执行完整的敏感性分析
    """
    analyzer = ParameterSensitivityAnalysis(tuning_results)

    # 1. 参数分布图
    dist_fig = analyzer.plot_parameter_distributions()
    plt.savefig('parameter_distributions.png')
    plt.close()

    # 2. 参数重要性
    importance_df = analyzer.calculate_parameter_importance()
    print("\n参数重要性分析:")
    print(importance_df)

    # 3. 参数重要性可视化
    imp_fig = analyzer.plot_parameter_importance()
    plt.savefig('parameter_importance.png')
    plt.close()

    # 4. 参数交互分析
    inter_fig = analyzer.analyze_interactions()
    plt.savefig('parameter_interactions.png')
    plt.close()

    return {
        'importance_df': importance_df,
        'figures': {
            'distributions': dist_fig,
            'importance': imp_fig,
            'interactions': inter_fig
        }
    }


def tune_nbeats_hyperparameters(Y_df, X_df, train_mask, param_grid):
    """
    NBeatsx模型超参数调优函数 - 修复版
    """




    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 计算总参数组合数
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    logger.info(f"总参数组合数: {total_combinations}")

    best_score = float('inf')
    best_params = None
    results = []

    # 生成所有参数组合
    param_keys = list(param_grid.keys())
    param_values = list(product(*[param_grid[key] for key in param_keys]))

    for idx, values in enumerate(param_values):
        current_params = dict(zip(param_keys, values))
        logger.info(f"\n测试参数组合 {idx + 1}/{total_combinations}")
        logger.info(f"当前参数: {current_params}")

        try:
            # 创建数据加载器
            train_loader = TimeSeriesLoader(
                model='nbeats',
                ts_dataset=ts_dataset,
                window_sampling_limit=current_params['window_sampling_limit'],
                offset=0,
                input_size=current_params['input_size'],
                output_size=current_params['output_size'],
                idx_to_sample_freq=1,
                batch_size=current_params['batch_size'],
                is_train_loader=True,
                shuffle=False
            )

            val_loader = TimeSeriesLoader(
                model='nbeats',
                ts_dataset=ts_dataset,
                window_sampling_limit=current_params['window_sampling_limit'],
                offset=0,
                input_size=current_params['input_size'],
                output_size=current_params['output_size'],
                idx_to_sample_freq=1,
                batch_size=current_params['batch_size'],
                is_train_loader=False,
                shuffle=False
            )

            # 创建include_var_dict
            include_var_dict = {
                'y': list(range(current_params['input_size'])),
                'Sal': list(range(current_params['input_size'])),
                'Temp_C': list(range(current_params['input_size']))
            }

            # 初始化模型
            model = Nbeats(
                input_size_multiplier=current_params['input_size']/current_params['output_size'],
                output_size=current_params['output_size'],
                shared_weights=False,
                initialization='glorot_normal',
                activation='relu',
                stack_types=['trend', 'seasonality', 'exogenous_tcn'],
                n_blocks=current_params['n_blocks'],
                n_layers=current_params['n_layers'],
                n_hidden=current_params['n_hidden'],
                n_harmonics=current_params['n_harmonics'],
                n_polynomials=current_params['n_polynomials'],
                x_s_n_hidden=0,
                exogenous_n_channels=2,
                include_var_dict=include_var_dict,
                t_cols=ts_dataset.t_cols,
                batch_normalization=True,
                dropout_prob_theta=current_params['dropout_prob_theta'],
                dropout_prob_exogenous=current_params['dropout_prob_exogenous'],
                learning_rate=current_params['learning_rate'],
                lr_decay=current_params['lr_decay'],
                n_lr_decay_steps=current_params['n_lr_decay_steps'],
                early_stopping=100,
                weight_decay=current_params['weight_decay'],
                l1_theta=current_params['l1_theta'],
                n_iterations=current_params['n_iterations'],
                loss=current_params['loss'],
                loss_hypar=current_params['loss_hypar'],
                val_loss=current_params['val_loss'],
                seasonality=current_params['seasonality'],
                random_seed=32
            )
            # 训练模型并评估
            history = model.fit(
                train_ts_loader=train_loader,
                val_ts_loader=val_loader,
                eval_steps=50
            )

            # 使用验证集评估模型
            y_true, y_hat, *_ = model.predict(ts_loader=val_loader)
            val_score = float(np.mean(np.abs(y_true - y_hat)))

            logger.info(f"验证集MAE: {val_score}")

            # 保存结果
            results.append({
                'params': current_params,
                'score': val_score,
                'status': 'success'
            })

            # 更新最佳参数
            if val_score < best_score:
                best_score = val_score
                best_params = current_params.copy()
                logger.info(f"找到新的最佳参数！得分: {val_score}")

        except Exception as e:
            logger.error(f"参数组合测试失败: {str(e)}")
            results.append({
            'params': current_params,
            'score': float('inf'),
            'status': 'failed'
            })

    # 改进的敏感性分析
    if results:
        try:
            # 将结果转换为DataFrame
            analysis_data = []
            for result in results:
                if result['status'] == 'success':
                    row = {}
                    # 处理每个参数，将复杂类型转换为字符串
                    for key, value in result['params'].items():
                        if isinstance(value, (list, tuple, dict)):
                            row[key] = str(value)
                        else:
                            row[key] = value
                    row['score'] = result['score']
                    analysis_data.append(row)

            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)

                # 计算相关性
                correlations = {}
                p_values = {}

                for column in analysis_df.columns:
                    if column != 'score':
                        # 确保列有多个不同的值
                        unique_values = analysis_df[column].nunique()
                        if unique_values > 1:
                            # 对非数值类型进行编码
                            if not np.issubdtype(analysis_df[column].dtype, np.number):
                                le = LabelEncoder()
                                values = le.fit_transform(analysis_df[column].astype(str))
                            else:
                                values = analysis_df[column].values

                            corr, p_value = spearmanr(values, analysis_df['score'])
                            correlations[column] = abs(corr)
                            p_values[column] = p_value
                        else:
                            logger.warning(f"参数 {column} 只有一个取值，跳过相关性计算")
                            correlations[column] = 0
                            p_values[column] = 1.0

                importance_df = pd.DataFrame({
                    'correlation': correlations,
                    'p_value': p_values
                }).sort_values('correlation', ascending=False)

                # 添加参数取值范围信息
                param_ranges = {}
                for column in analysis_df.columns:
                    if column != 'score':
                        unique_values = analysis_df[column].unique()
                        param_ranges[column] = f"{len(unique_values)} 个值: {unique_values.tolist()}"

                sensitivity_results = {
                    'importance_df': importance_df,
                    'param_ranges': param_ranges,
                    'analysis_df': analysis_df
                }
            else:
                sensitivity_results = None
                logger.warning("没有成功的训练结果用于分析")
        except Exception as e:
            logger.error(f"敏感性分析失败: {str(e)}")
            logger.error(traceback.format_exc())
            sensitivity_results = None
    else:
        sensitivity_results = None

    return best_params, best_score, results, sensitivity_results


# 更新测试参数网格，使用更保守的参数范围
test_param_grid = {
    'input_size': [2,4,8,12],
    'output_size': [2],
    'window_sampling_limit': [7259],
    'n_blocks': [[3, 3, 3], [4, 4, 4]],
    'n_layers': [[1, 1, 1]],
    'n_hidden': [
        [[200, 200, 200], [200, 200, 200], [200, 200, 200]],
        [[300, 300, 300], [300, 300, 300], [300, 300, 300]]
    ],
    'n_harmonics': [2, 3],
    'n_polynomials': [2],
    'seasonality': [24],
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.0001],
    'lr_decay': [0.9],
    'n_lr_decay_steps': [1],
    'dropout_prob_theta': [0.1, 0.2],
    'dropout_prob_exogenous': [0],
    'weight_decay': [0],
    'l1_theta': [0],
    'n_iterations': [500],
    'loss': ['MAE'],
    'loss_hypar': [0.5],
    'val_loss': ['MAE']
}

# 使用示例：
"""
try:
    best_params, best_score, all_results, sensitivity_results = tune_nbeats_hyperparameters(
        Y_df, X_df, train_mask_1, test_param_grid
    )

    print("\n最佳参数组合:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"\n最佳得分: {best_score}")

    if sensitivity_results is not None:
        print("\n参数重要性分析:")
        print(sensitivity_results['importance_df'])

except Exception as e:
    print(f"运行过程中发生错误: {str(e)}")
"""

def create_include_var_dict(input_size):
    """
    根据输入窗口大小创建包含滞后的字典
    """
    lags = list(range(input_size))
    return {
        'y': lags,
        'Sal': lags,
        'Temp_C': lags
    }
# 示例使用:


try:
    best_params, best_score, all_results, sensitivity_results = tune_nbeats_hyperparameters(
        Y_df, X_df, train_mask_1, test_param_grid
    )

    print("\n最佳参数组合:")
    if best_params:
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"\n最佳得分: {best_score}")

    if sensitivity_results:
        print("\n参数重要性分析:")
        print(sensitivity_results['importance_df'])

        print("\n参数取值范围:")
        for param, range_info in sensitivity_results['param_ranges'].items():
            print(f"{param}: {range_info}")

        # 打印测试的参数组合数量
        print(f"\n成功测试的参数组合数量: {len(sensitivity_results['analysis_df'])}")
    else:
        print("\n无法进行参数重要性分析")

except Exception as e:
    print(f"运行过程中发生错误: {str(e)}")
    traceback.print_exc()