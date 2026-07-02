
"""
浮游生物丰度预测入口。

该脚本由外部调度器定时调用，每次只执行一轮预测，不在脚本内部常驻循环。
当前稳定实现包含两条链路：
1. 小时预测：使用过去 12 小时数据预测未来 1 小时。
2. 日预测：在指定小时使用过去 7 个完整自然日数据预测当前自然日。

文件下方保留了之前探索阶段代码作为参考，但正常执行 `python start_main.py`
会在本入口结束，不会继续执行旧代码。
"""

import os as _os
import json as _json
import sys as _sys
import traceback as _traceback
from datetime import datetime as _datetime, timedelta as _timedelta

import pandas as _pd

from all_function import (
    get_currenttime_before_12hour_fun as _get_currenttime_before_12hour_fun,
)
from h12_model_train import read_12h_train_tab_to_train_model_fun as _read_12h_train_tab_to_train_model_fun
from h12_model_train import read_7d_train_tab_to_train_model_fun as _read_7d_train_tab_to_train_model_fun
from nbeats.nbeats_model import d7_predict_next_1d_points as _d7_predict_next_1d_points
from nbeats.nbeats_model import h12_predict_next_1h_points as _h12_predict_next_1h_points
from pymysql_data import create_mysql as _create_mysql
from pymysql_data import mysqlConnect as _mysqlConnect
from pymysql_data import mysql_column_type as _mysql_column_type
from pymysql_data import mysqlinsert_12h_predict as _mysqlinsert_12h_predict



_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(_SCRIPT_DIR)


#### 数据库配置。
DATA_IP = "192.168.2.82" # 数据库 IP
DATA_PORT = 3306 # 数据库端口
DATA_USER = "root" # 数据库用户
DATA_PASSWORD = "root" # 数据库密码
DATA_NAME = "szsw_plant" # 数据库名称


#### 表配置。
SOURCE_AVG_TABLE = "tab_biology_avg"   # 雷工均值表，包含小时均值和日均值，按 SummaryInterval 区分
WATER_QUALITY_TABLE = "tab_waterqualityrecord" # 水质记录表，按 StatusTime 对齐协变量
TRAIN_12H_TABLE = "tab_train_12h"   # 小时训练表，存储小时预测训练数据
PREDICT_12H_RECORD_TABLE = "tab_predict_12h_record"  # 小时预测记录表，存储小时预测结果
PREDICT_12H_INPUT_TABLE = "tab_predict_12h_input_record" # 小时预测输入留痕表，存储小时预测使用的历史输入窗口数据
TRAIN_7D_TABLE = "tab_train_7d" # 日训练表，存储日预测训练数据
PREDICT_7D_RECORD_TABLE = "tab_predict_7d_record" # 日预测记录表，存储日预测结果
PREDICT_7D_INPUT_TABLE = "tab_predict_7d_input_record" # 日预测输入留痕表，存储日预测使用的历史输入窗口数据


#### 预测物种和协变量。
NEED_PREDICT_BIO = ["copepodadensity"] # 需要预测的浮游生物物种
CONCOMITANT_VARIABLES = ["Temp", "PH"] # 协变量 温度 和 pH
BIOLOGY_DEVICE_ID = 7 # 生物均值表查询设备
WATER_QUALITY_DEVICE_ID = 7 # 水质表查询设备


#### 模型和运行配置。
MODEL_DIR = "saved_models"   # 模型保存目录
ENABLE_TEST_TIME_OVERRIDE = False # 是否启用测试时间覆盖，启用后使用 TEST_CURRENT_TIME 作为当前时间
TEST_CURRENT_TIME = "2026-06-30 15:00:00" # 测试时间覆盖，启用后使用 TEST_CURRENT_TIME 作为当前时间
ENABLE_7D_PREDICTION = True # 是否启用日预测
DAILY_PREDICTION_HOUR = 13 # 日预测执行小时，0 表示在每天的 0 点执行日预测
HOURLY_MODEL_RETRAIN_HOUR = 23 # 小时模型每日固定重训小时，默认在每天最后一次小时调度时重训
DAILY_MODEL_RETRAIN_HOUR = DAILY_PREDICTION_HOUR # 日模型每日固定重训小时，默认与日预测执行小时一致
MIN_HOURLY_SOURCE_POINTS = 24 # 小时预测所需的最少源数据点数，当前整点前 24 小时真实数据不足则跳过小时预测
MIN_DAILY_SOURCE_DAYS = 7 # 日预测所需的最少源数据天数，最近 7 个完整自然日真实数据不足则跳过日预测


#### 建表字段。
TABLE_CREATE_COLUMNS = ["SnapTime"] + NEED_PREDICT_BIO + CONCOMITANT_VARIABLES

PREDICT_12H_INPUT_COLUMNS = ["PredictSnapTime"] + TABLE_CREATE_COLUMNS 
PREDICT_7D_INPUT_COLUMNS = ["PredictSnapTime"] + TABLE_CREATE_COLUMNS


def _get_current_hour():
    """获取当前调度时间，并向下归整到整点。"""
    if ENABLE_TEST_TIME_OVERRIDE:
        # TEST_CURRENT_TIME = _datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return _datetime.strptime(TEST_CURRENT_TIME, "%Y-%m-%d %H:%M:%S")
    return _datetime.now().replace(minute=0, second=0, microsecond=0)


def _ensure_required_tables():
    """创建当前稳定链路需要的训练表和预测记录表。"""
    _create_mysql(TABLE_CREATE_COLUMNS, DATA_IP, DATA_PORT, DATA_USER, DATA_PASSWORD, DATA_NAME, TRAIN_12H_TABLE)
    _create_mysql(
        TABLE_CREATE_COLUMNS,
        DATA_IP,
        DATA_PORT,
        DATA_USER,
        DATA_PASSWORD,
        DATA_NAME,
        PREDICT_12H_RECORD_TABLE,
    )
    _create_mysql(
        PREDICT_12H_INPUT_COLUMNS,
        DATA_IP,
        DATA_PORT,
        DATA_USER,
        DATA_PASSWORD,
        DATA_NAME,
        PREDICT_12H_INPUT_TABLE,
    )
    _create_mysql(TABLE_CREATE_COLUMNS, DATA_IP, DATA_PORT, DATA_USER, DATA_PASSWORD, DATA_NAME, TRAIN_7D_TABLE)
    _create_mysql(
        TABLE_CREATE_COLUMNS,
        DATA_IP,
        DATA_PORT,
        DATA_USER,
        DATA_PASSWORD,
        DATA_NAME,
        PREDICT_7D_RECORD_TABLE,
    )
    _create_mysql(
        PREDICT_7D_INPUT_COLUMNS,
        DATA_IP,
        DATA_PORT,
        DATA_USER,
        DATA_PASSWORD,
        DATA_NAME,
        PREDICT_7D_INPUT_TABLE,
    )


def _connect_database():
    return _mysqlConnect(DATA_IP, DATA_PORT, DATA_USER, DATA_PASSWORD, DATA_NAME)


def _ensure_columns_exist(conn, table_name, columns):
    """只给目标表补缺失字段，不修改任何源数据表。"""
    cursor = conn.cursor()
    try:
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        existing_columns = {row[0] for row in cursor.fetchall()}
        for column_name in columns:
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN `{column_name}` {_mysql_column_type(column_name)}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def _ensure_target_table_columns(conn):
    """确保训练表、预测表、输入留痕表包含当前业务需要的字段。"""
    _ensure_columns_exist(conn, TRAIN_12H_TABLE, TABLE_CREATE_COLUMNS)
    _ensure_columns_exist(conn, PREDICT_12H_RECORD_TABLE, TABLE_CREATE_COLUMNS)
    _ensure_columns_exist(conn, PREDICT_12H_INPUT_TABLE, PREDICT_12H_INPUT_COLUMNS)
    _ensure_columns_exist(conn, TRAIN_7D_TABLE, TABLE_CREATE_COLUMNS)
    _ensure_columns_exist(conn, PREDICT_7D_RECORD_TABLE, TABLE_CREATE_COLUMNS)
    _ensure_columns_exist(conn, PREDICT_7D_INPUT_TABLE, PREDICT_7D_INPUT_COLUMNS)


def _model_path(bio_name, model_suffix=""):
    return _os.path.join(MODEL_DIR, f"{bio_name}{model_suffix}_nbeats_model.pkl")


def _model_params_path(bio_name, model_suffix=""):
    return _os.path.join(MODEL_DIR, f"{bio_name}{model_suffix}_model_params.json")


def _model_uses_current_variables(bio_name, model_suffix=""):
    """检查模型参数中的协变量是否与当前配置一致。"""
    params_path = _model_params_path(bio_name, model_suffix)
    if not _os.path.exists(params_path):
        return False
    try:
        with open(params_path, 'r', encoding='utf-8') as params_file:
            model_config = _json.load(params_file)
        include_var_dict = model_config.get('include_var_dict', {})
        model_variables = [key for key in include_var_dict.keys() if key != 'y']
        return model_variables == CONCOMITANT_VARIABLES
    except Exception:
        return False


def _validate_model_files(model_suffix=""):
    """预测前校验目标物种模型文件是否存在且协变量配置匹配。"""
    missing_models = []
    incompatible_models = []
    for bio_name in NEED_PREDICT_BIO:
        model_path = _model_path(bio_name, model_suffix)
        if not _os.path.exists(model_path):
            missing_models.append(model_path)
        elif not _model_uses_current_variables(bio_name, model_suffix):
            incompatible_models.append(model_path)
    if missing_models:
        raise FileNotFoundError("缺少模型文件: " + ", ".join(missing_models))
    if incompatible_models:
        raise RuntimeError("模型协变量配置已失效，需要重新训练: " + ", ".join(incompatible_models))


def _has_missing_model_files(model_suffix=""):
    """判断是否存在缺失或协变量配置失效的目标物种模型文件。"""
    return any(
        not _os.path.exists(_model_path(bio_name, model_suffix))
        or not _model_uses_current_variables(bio_name, model_suffix)
        for bio_name in NEED_PREDICT_BIO
    )


def _ensure_hourly_models_exist(conn):
    """小时模型缺失时，先尝试训练；训练失败则优雅跳过小时预测。"""
    if not _has_missing_model_files():
        return True

    print("小时模型文件缺失，开始先训练小时模型")
    try:
        trained = _read_12h_train_tab_to_train_model_fun(conn, TRAIN_12H_TABLE, NEED_PREDICT_BIO, CONCOMITANT_VARIABLES)
        if trained is False:
            print("警告：小时模型文件缺失，且训练数据不足，本次跳过小时预测")
            return False
        _validate_model_files()
        return True
    except Exception as error:
        print(f"警告：小时模型缺失时兜底训练失败，本次跳过小时预测。错误信息: {error}")
        return False


def _ensure_daily_models_exist(conn):
    """日模型缺失或协变量失效时，先尝试训练；训练失败则优雅跳过日预测。"""
    if not _has_missing_model_files(model_suffix="_d7"):
        return True

    print("7d日模型文件缺失或协变量配置失效，开始先训练日模型")
    try:
        trained = _read_7d_train_tab_to_train_model_fun(conn, TRAIN_7D_TABLE, NEED_PREDICT_BIO, CONCOMITANT_VARIABLES)
        if trained is False:
            print("警告：7d日模型训练数据不足，本次跳过日预测")
            return False
        _validate_model_files(model_suffix="_d7")
        return True
    except Exception as error:
        print(f"警告：7d日模型兜底训练失败，本次跳过日预测。错误信息: {error}")
        return False


def _should_retrain_today(current_time, model_suffix=""):
    """任一目标物种模型文件早于当前日期时，触发每日重训。"""
    current_date = current_time.date()
    for bio_name in NEED_PREDICT_BIO:
        model_path = _model_path(bio_name, model_suffix)
        if not _os.path.exists(model_path):
            return True
        if not _model_uses_current_variables(bio_name, model_suffix):
            print(f"{bio_name} 模型协变量配置与当前 {CONCOMITANT_VARIABLES} 不一致，需要重训")
            return True
        model_date = _datetime.fromtimestamp(_os.path.getmtime(model_path)).date()
        if model_date < current_date:
            print(f"{bio_name} 模型日期为 {model_date}，早于当前日期 {current_date}，需要重训")
            return True
    return False


def _should_retrain_at_fixed_hour(current_time, retrain_hour, model_suffix=""):
    """只在固定小时执行每日重训判断，避免每个整点都触发重训逻辑。"""
    if current_time.hour != retrain_hour:
        return False
    return _should_retrain_today(current_time, model_suffix)


def _predict_all_species(history_by_species, target_time, predict_func):
    """所有物种都预测成功后，返回一条完整预测记录。"""
    predict_row = {"SnapTime": target_time}

    for bio_name in NEED_PREDICT_BIO:
        if bio_name not in history_by_species:
            raise KeyError(f"查询结果中缺少物种 {bio_name}")

        historical_data = history_by_species[bio_name]
        predictions = predict_func(historical_data, bio_name, model_dir=MODEL_DIR)
        if len(predictions) == 0:
            raise ValueError(f"{bio_name} 模型未返回预测值")

        raw_value = float(predictions[-1])
        predict_value = round(raw_value, 2) if raw_value >= 0 else 0
        predict_row[bio_name] = predict_value
        print(f"预测时间 {target_time}，物种 {bio_name}，预测值 {predict_value}")

    return predict_row


def _insert_row_if_missing(conn, table_name, row_data, unique_keys=None):
    """按指定字段去重插入一条记录，默认按 SnapTime 去重。"""
    if unique_keys is None:
        unique_keys = ["SnapTime"]

    columns = list(row_data.keys())
    values = list(row_data.values())
    placeholders = ", ".join(["%s"] * len(values))
    column_sql = ", ".join(columns)
    unique_sql = " AND ".join([f"{key} = %s" for key in unique_keys])
    unique_values = tuple(row_data[key] for key in unique_keys)
    sql = (
        f"INSERT INTO {table_name} ({column_sql}) "
        f"SELECT {placeholders} FROM DUAL "
        f"WHERE NOT EXISTS(SELECT 1 FROM {table_name} WHERE {unique_sql})"
    )
    cursor = conn.cursor()
    try:
        cursor.execute(sql, tuple(values) + unique_values)
        affected_rows = cursor.rowcount
        conn.commit()
        return affected_rows
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def _record_exists(conn, table_name, snap_time):
    """检查某个预测时刻是否已经存在记录。"""
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT 1 FROM {table_name} WHERE SnapTime = %s LIMIT 1", (snap_time,))
        return cursor.fetchone() is not None
    finally:
        cursor.close()


def _get_latest_snap_time(conn, table_name):
    """读取目标训练表中最新的 SnapTime，用于增量同步。"""
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT MAX(SnapTime) FROM {table_name}")
        result = cursor.fetchone()
        if not result or result[0] is None:
            return None
        latest_time = _pd.to_datetime(result[0], errors='coerce')
        if _pd.isna(latest_time):
            return None
        return latest_time.to_pydatetime()
    finally:
        cursor.close()


def _normalize_time_window(df, time_column, target_index, value_columns):
    """按目标时间轴补齐数值列，缺失值使用插值和前后填充。"""
    if time_column not in df.columns:
        df = _pd.DataFrame(columns=[time_column] + value_columns)
    df[time_column] = _pd.to_datetime(df[time_column], errors='coerce')
    df = df.dropna(subset=[time_column]).sort_values(time_column)

    for column_name in value_columns:
        if column_name not in df.columns:
            df[column_name] = 0.0

    if df.empty:
        window_df = _pd.DataFrame(index=target_index, columns=value_columns)
    else:
        window_df = df[[time_column] + value_columns].copy()
        window_df = window_df.drop_duplicates(subset=[time_column], keep='last')
        window_df = window_df.set_index(time_column).reindex(target_index)

    for column_name in value_columns:
        window_df[column_name] = _pd.to_numeric(window_df[column_name], errors='coerce')
    return window_df.interpolate(method='linear', limit_direction='both').ffill().bfill().fillna(0.0)


def _read_water_quality_means(conn, target_index, freq):
    """从水质表只读查询，并按小时或自然日聚合协变量均值。"""
    if len(target_index) == 0:
        return _pd.DataFrame(index=target_index, columns=CONCOMITANT_VARIABLES).fillna(0.0)

    start_time = target_index.min()
    if freq == 'H':
        end_time = target_index.max() + _timedelta(hours=1)
    else:
        end_time = target_index.max() + _timedelta(days=1)

    columns_sql = ", ".join(["StatusTime"] + CONCOMITANT_VARIABLES)
    sql = f"""SELECT {columns_sql} FROM {WATER_QUALITY_TABLE}
    WHERE StatusTime >= %s AND StatusTime < %s AND DeviceID=%s"""
    df = _pd.read_sql(sql, conn, params=(start_time, end_time, WATER_QUALITY_DEVICE_ID))
    if 'StatusTime' not in df.columns:
        df = _pd.DataFrame(columns=['StatusTime'] + CONCOMITANT_VARIABLES)
    df['StatusTime'] = _pd.to_datetime(df['StatusTime'], errors='coerce')
    df = df.dropna(subset=['StatusTime'])
    for column_name in CONCOMITANT_VARIABLES:
        if column_name not in df.columns:
            df[column_name] = 0.0
        df[column_name] = _pd.to_numeric(df[column_name], errors='coerce')

    if df.empty:
        grouped = _pd.DataFrame(columns=['SnapTime'] + CONCOMITANT_VARIABLES)
    else:
        df['SnapTime'] = df['StatusTime'].dt.floor(freq)
        grouped = df.groupby('SnapTime', as_index=False)[CONCOMITANT_VARIABLES].mean()

    return _normalize_time_window(grouped, 'SnapTime', target_index, CONCOMITANT_VARIABLES)


def _count_recent_hourly_points(conn, current_time):
    """统计当前整点前 24 小时源表中真实存在的小时记录数。"""
    start_time = current_time - _timedelta(hours=23)
    sql = f"""SELECT COUNT(DISTINCT SnapTime) FROM {SOURCE_AVG_TABLE}
    WHERE SnapTime >= %s AND SnapTime <= %s AND SummaryInterval=60 AND DeviceID=%s"""
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (start_time, current_time, BIOLOGY_DEVICE_ID))
        result = cursor.fetchone()
        return int(result[0]) if result and result[0] is not None else 0
    finally:
        cursor.close()


def _build_hourly_history(conn, before_12h_times):
    """构建小时预测输入：生物值来自生物表，协变量来自水质表按小时均值。"""
    target_times = sorted(_pd.to_datetime(before_12h_times))
    target_index = _pd.DatetimeIndex(target_times)
    start_time = target_index.min()
    end_time = target_index.max()

    columns_sql = ", ".join(["SnapTime"] + NEED_PREDICT_BIO)
    find_sql = f"""SELECT {columns_sql} FROM {SOURCE_AVG_TABLE}
    WHERE SnapTime >= %s AND SnapTime <= %s AND SummaryInterval=60 AND DeviceID=%s"""
    biology_df = _pd.read_sql(find_sql, conn, params=(start_time, end_time, BIOLOGY_DEVICE_ID))
    biology_window = _normalize_time_window(biology_df, 'SnapTime', target_index, NEED_PREDICT_BIO)
    water_window = _read_water_quality_means(conn, target_index, freq='H')

    history_by_species = {}
    for bio_name in NEED_PREDICT_BIO:
        species_history = {'y': biology_window[bio_name].astype(float).tolist()}
        for variable_name in CONCOMITANT_VARIABLES:
            species_history[variable_name] = water_window[variable_name].astype(float).tolist()
        history_by_species[bio_name] = species_history

    return history_by_species


def _write_hourly_window_to_predict_input(conn, history_by_species, current_time, target_time):
    """把本次预测使用的 12 小时输入窗口写入预测输入留痕表。"""
    for index in range(12):
        snap_time = current_time - _timedelta(hours=11 - index)
        row_data = {'PredictSnapTime': target_time, 'SnapTime': snap_time}
        for bio_name in NEED_PREDICT_BIO:
            row_data[bio_name] = history_by_species[bio_name]['y'][index]
        for variable_name in CONCOMITANT_VARIABLES:
            row_data[variable_name] = history_by_species[NEED_PREDICT_BIO[0]][variable_name][index]
        _insert_row_if_missing(
            conn,
            PREDICT_12H_INPUT_TABLE,
            row_data,
            unique_keys=["PredictSnapTime", "SnapTime"],
        )


def _write_recent_source_hours_to_train(conn):
    """增量同步小时生物历史数据到训练表，协变量按水质小时均值对齐。"""
    latest_train_time = _get_latest_snap_time(conn, TRAIN_12H_TABLE)
    columns_sql = ", ".join(["SnapTime"] + NEED_PREDICT_BIO)
    find_sql = f"""SELECT {columns_sql} FROM {SOURCE_AVG_TABLE}
    WHERE SummaryInterval=60 AND DeviceID=%s"""
    params = [BIOLOGY_DEVICE_ID]
    if latest_train_time is not None:
        find_sql += " AND SnapTime > %s"
        params.append(latest_train_time)
    df = _pd.read_sql(find_sql, conn, params=tuple(params))
    if 'SnapTime' not in df.columns:
        return
    df['SnapTime'] = _pd.to_datetime(df['SnapTime'], errors='coerce')
    df = df.dropna(subset=['SnapTime']).sort_values('SnapTime')
    if df.empty:
        print("小时训练表同步跳过：没有新的小时源数据")
        return

    target_index = _pd.DatetimeIndex(df['SnapTime'].drop_duplicates().sort_values())
    water_window = _read_water_quality_means(conn, target_index, freq='H')

    inserted_count = 0
    for _, source_row in df.iterrows():
        snap_time = source_row['SnapTime']
        row_data = {'SnapTime': snap_time.to_pydatetime()}
        for bio_name in NEED_PREDICT_BIO:
            value = _pd.to_numeric(_pd.Series([source_row.get(bio_name)]), errors='coerce').iloc[0]
            row_data[bio_name] = 0.0 if _pd.isna(value) else float(value)
        for variable_name in CONCOMITANT_VARIABLES:
            row_data[variable_name] = float(water_window.loc[snap_time, variable_name])
        inserted_count += _insert_row_if_missing(conn, TRAIN_12H_TABLE, row_data)
    print(f"小时训练表增量同步完成：查询到新小时数据 {len(df)} 条，本次新增 {inserted_count} 条到 {TRAIN_12H_TABLE}")


def _build_daily_history(conn, target_day):
    """读取 T-7 到 T-1 的日级预测输入，协变量按水质自然日均值对齐。"""
    start_day = target_day - _timedelta(days=7)
    columns_sql = ", ".join(["SnapTime"] + NEED_PREDICT_BIO)
    find_sql = f"""SELECT {columns_sql} FROM {SOURCE_AVG_TABLE}
    WHERE SnapTime >= %s AND SnapTime < %s AND SummaryInterval=1 AND SummaryIntervalUnit='天' AND DeviceID=%s"""
    # print(find_sql) # 测试查询输出

    df = _pd.read_sql(find_sql, conn, params=(start_day, target_day, BIOLOGY_DEVICE_ID))
    if 'SnapTime' in df.columns:
        df['SnapTime'] = _pd.to_datetime(df['SnapTime'], errors='coerce')
        df = df.dropna(subset=['SnapTime'])
        df['Day'] = df['SnapTime'].dt.date

    history_by_species = {
        bio_name: {'y': []}
        for bio_name in NEED_PREDICT_BIO
    }
    for bio_name in NEED_PREDICT_BIO:
        for variable_name in CONCOMITANT_VARIABLES:
            history_by_species[bio_name][variable_name] = []

    daily_rows = []
    source_days = set(df['Day'].dropna().tolist()) if 'Day' in df.columns else set()

    last_history_day = (target_day - _timedelta(days=1)).date()
    if 'Day' in df.columns and not df.empty:
        full_daily_index = _pd.date_range(start_day, last_history_day, freq='D')
        water_window = _read_water_quality_means(conn, full_daily_index, freq='D')

    if 'Day' in df.columns and not df.empty:
        for day, day_df in df.groupby('Day'):
            if day > last_history_day:
                continue
            day_snap_time = _datetime.combine(day, _datetime.min.time())
            daily_row = {'SnapTime': day_snap_time}
            for bio_name in NEED_PREDICT_BIO:
                value_series = _pd.to_numeric(day_df[bio_name], errors='coerce').dropna() if bio_name in day_df.columns else _pd.Series(dtype=float)
                value = value_series.iloc[-1] if not value_series.empty else 0.0
                daily_row[bio_name] = 0.0 if _pd.isna(value) else float(value)
            for variable_name in CONCOMITANT_VARIABLES:
                daily_row[variable_name] = float(water_window.loc[day_snap_time, variable_name])
            daily_rows.append(daily_row)

    # 日预测目标日为 T，模型输入严格使用不包含 T 当天的 7 天窗口：T-7 到 T-1。
    target_index = _pd.date_range(start_day, last_history_day, freq='D')
    if daily_rows:
        window_df = _pd.DataFrame(daily_rows)
    else:
        window_df = _pd.DataFrame(columns=TABLE_CREATE_COLUMNS)
    window_df['SnapTime'] = _pd.to_datetime(window_df['SnapTime'], errors='coerce')
    window_df = window_df.dropna(subset=['SnapTime']).drop_duplicates(subset=['SnapTime'], keep='last')
    for column_name in TABLE_CREATE_COLUMNS:
        if column_name == 'SnapTime':
            continue
        if column_name not in window_df.columns:
            window_df[column_name] = 0.0
    window_df = window_df.set_index('SnapTime').reindex(target_index)
    for column_name in TABLE_CREATE_COLUMNS:
        if column_name == 'SnapTime':
            continue
        window_df[column_name] = _pd.to_numeric(window_df[column_name], errors='coerce')
    window_df = window_df.interpolate(method='linear', limit_direction='both').ffill().bfill().fillna(0.0)

    window_df = window_df.reset_index().rename(columns={'index': 'SnapTime'})
    for _, daily_row in window_df.iterrows():
        for bio_name in NEED_PREDICT_BIO:
            history_by_species[bio_name]['y'].append(daily_row[bio_name])
            for variable_name in CONCOMITANT_VARIABLES:
                history_by_species[bio_name][variable_name].append(daily_row[variable_name])

    # print(f"日级历史数据: {history_by_species}") # 测试输出
    return history_by_species, source_days


def _write_daily_train_rows(conn, daily_rows):
    """把日级历史数据写入 7d 训练表，已存在的日期不重复插入。"""
    inserted_count = 0
    for row_data in daily_rows:
        inserted_count += _insert_row_if_missing(conn, TRAIN_7D_TABLE, row_data)
    print(f"7d训练表增量同步完成：查询到新日级记录 {len(daily_rows)} 天，本次新增 {inserted_count} 天到 {TRAIN_7D_TABLE}")


def _write_incremental_daily_source_to_train(conn, target_day):
    """增量同步日级生物数据到 7d 训练表，协变量按水质自然日均值对齐。"""
    latest_train_time = _get_latest_snap_time(conn, TRAIN_7D_TABLE)
    columns_sql = ", ".join(["SnapTime"] + NEED_PREDICT_BIO)
    find_sql = f"""SELECT {columns_sql} FROM {SOURCE_AVG_TABLE}
    WHERE SnapTime < %s AND SummaryInterval=1 AND SummaryIntervalUnit='天' AND DeviceID=%s"""
    params = [target_day, BIOLOGY_DEVICE_ID]
    if latest_train_time is not None:
        find_sql += " AND SnapTime > %s"
        params.append(latest_train_time)

    df = _pd.read_sql(find_sql, conn, params=tuple(params))
    if 'SnapTime' not in df.columns:
        return
    df['SnapTime'] = _pd.to_datetime(df['SnapTime'], errors='coerce')
    df = df.dropna(subset=['SnapTime']).sort_values('SnapTime')
    if df.empty:
        print("7d训练表同步跳过：没有新的日级源数据")
        return

    target_index = _pd.DatetimeIndex(df['SnapTime'].drop_duplicates().sort_values())
    water_window = _read_water_quality_means(conn, target_index, freq='D')

    daily_rows = []
    for _, source_row in df.iterrows():
        snap_time = source_row['SnapTime']
        row_data = {'SnapTime': snap_time.to_pydatetime()}
        for bio_name in NEED_PREDICT_BIO:
            value = _pd.to_numeric(_pd.Series([source_row.get(bio_name)]), errors='coerce').iloc[0]
            row_data[bio_name] = 0.0 if _pd.isna(value) else float(value)
        for variable_name in CONCOMITANT_VARIABLES:
            row_data[variable_name] = float(water_window.loc[snap_time, variable_name])
        daily_rows.append(row_data)

    _write_daily_train_rows(conn, daily_rows)


def _write_daily_window_to_predict_input(conn, history_by_species, target_day):
    """把本次日预测使用的 7 天输入窗口写入预测输入留痕表，窗口为 T-7 到 T-1，不包含目标日 T。"""
    start_day = target_day - _timedelta(days=7)
    for index in range(7):
        snap_time = start_day + _timedelta(days=index)
        row_data = {'PredictSnapTime': target_day, 'SnapTime': snap_time}
        for bio_name in NEED_PREDICT_BIO:
            row_data[bio_name] = history_by_species[bio_name]['y'][index]
        for variable_name in CONCOMITANT_VARIABLES:
            row_data[variable_name] = history_by_species[NEED_PREDICT_BIO[0]][variable_name][index]
        _insert_row_if_missing(
            conn,
            PREDICT_7D_INPUT_TABLE,
            row_data,
            unique_keys=["PredictSnapTime", "SnapTime"],
        )


def _run_12h_pipeline(conn, current_time):
    """执行 12h -> 1h 小时预测链路。"""
    target_time = current_time + _timedelta(hours=1)
    print(f"小时预测目标时间: {target_time}")

    real_hourly_points = _count_recent_hourly_points(conn, current_time)
    if real_hourly_points < MIN_HOURLY_SOURCE_POINTS:
        print(f"小时预测跳过：当前时间前24小时真实数据只有 {real_hourly_points} 条，少于 {MIN_HOURLY_SOURCE_POINTS} 条")
        return

    before_12h_times = _get_currenttime_before_12hour_fun(current_time)
    history_by_species = _build_hourly_history(conn, before_12h_times)

    _write_recent_source_hours_to_train(conn)
    conn.commit()
    if not _ensure_hourly_models_exist(conn):
        return

    print(f"小时历史数据查询结果: {history_by_species}") # 测试输出
    predict_row = _predict_all_species(history_by_species, target_time, _h12_predict_next_1h_points)

    _mysqlinsert_12h_predict(
        conn,
        PREDICT_12H_RECORD_TABLE,
        predict_row.values(),
        predict_row.keys(),
    )

    _write_hourly_window_to_predict_input(conn, history_by_species, current_time, target_time)

    if _should_retrain_at_fixed_hour(current_time, HOURLY_MODEL_RETRAIN_HOUR):
        print("开始小时模型每日重训")
        _read_12h_train_tab_to_train_model_fun(conn, TRAIN_12H_TABLE, NEED_PREDICT_BIO, CONCOMITANT_VARIABLES)
    else:
        print(f"当前不是小时模型固定重训小时 {HOURLY_MODEL_RETRAIN_HOUR}，或今日小时模型已训练，无需重训")


def _run_7d_pipeline(conn, current_time):
    """按配置执行 7d -> 1d 日预测链路。"""
    if not ENABLE_7D_PREDICTION:
        print("7d日预测未启用")
        return
    if current_time.hour != DAILY_PREDICTION_HOUR:
        print(f"当前小时 {current_time.hour} 不是日预测执行小时 {DAILY_PREDICTION_HOUR}，跳过7d日预测")
        return

    target_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"日预测目标日期: {target_day}")
    if _record_exists(conn, PREDICT_7D_RECORD_TABLE, target_day):
        print(f"{target_day} 的日预测记录已存在，跳过7d日预测")
        return

    history_by_species, source_days = _build_daily_history(conn, target_day)
    real_daily_days = 0
    for day_offset in range(7):
        day = (target_day - _timedelta(days=7 - day_offset)).date()
        if day in source_days:
            real_daily_days += 1
    if real_daily_days < MIN_DAILY_SOURCE_DAYS:
        print(f"日预测跳过：最近7个完整自然日真实数据只有 {real_daily_days} 天，少于 {MIN_DAILY_SOURCE_DAYS} 天")
        return

    _write_incremental_daily_source_to_train(conn, target_day)

    if _should_retrain_at_fixed_hour(current_time, DAILY_MODEL_RETRAIN_HOUR, model_suffix="_d7"):
        print("开始7d日模型每日重训")
        trained = _read_7d_train_tab_to_train_model_fun(conn, TRAIN_7D_TABLE, NEED_PREDICT_BIO, CONCOMITANT_VARIABLES)
        if not trained:
            raise RuntimeError("7d日模型训练数据不足，无法完成日预测")

    if not _ensure_daily_models_exist(conn):
        return

    _validate_model_files(model_suffix="_d7")
    predict_row = _predict_all_species(history_by_species, target_day, _d7_predict_next_1d_points)
    _insert_row_if_missing(conn, PREDICT_7D_RECORD_TABLE, predict_row)
    _write_daily_window_to_predict_input(conn, history_by_species, target_day)


def _run_once():
    current_time = _get_current_hour() # 当前整点
    # current_time = _datetime.strptime("2024-08-21 23:00:00", "%Y-%m-%d %H:%M:%S") # 测试用

    print(f"当前整点: {current_time}")

    _ensure_required_tables()
    conn = _connect_database()
    try:
        _ensure_target_table_columns(conn)
        _run_12h_pipeline(conn, current_time)
        _run_7d_pipeline(conn, current_time)
    finally:
        conn.close()


def _main():
    try:
        _run_once()
        return 0
    except Exception:
        print("本次预测失败，未写入部分预测结果")
        _traceback.print_exc()
        return 1


if __name__ == "__main__":
    _sys.exit(_main())

r"""
以下为之前探索阶段代码，仅作参考，不参与当前正式入口执行。

'''
补充：每天训练一次
但是在模型训练的过程中，我们训练的数据取的越长越好,对于小时预测-小于24个数据不预测，对于天预测-小于7个数据不预测
1、对于小时预测，取过去24小时，预测接下来的1小时(12h预测1h，24h预测2h)
2、对于天预测，取过去7天，预测接下来的1天
3、预测间隔1h模型训练一次，同时预测一次


核心步骤：
1、查询雷工的均值、和值表
2、拷贝进入自己的实际值表（保证小时值连续且全有，没有直接补0）
3、根据实际表训练模型，得到下一个小时的预测值，写入预测表（可能会轮动，直接预测接下来24小时的值）
4、将预测值写入预测表格
特别注意：预测的是连续时间序列，中间如果有缺少值，需要插值法（直接补0）
'''
import math
import traceback

'''
驱动说明：
程序驱动，以整点时刻进行预测一次，1小时预测一次物种，看是否能同时预测所有物种，如果不能，则每小时都要训练多个模型
关于数据表说明：

'''
from pymysql_data import *
import time,os
from nbeats.nbeats_model import h12_predict_next_1h_points
from h12_model_train import read_12h_train_tab_to_train_model_fun
from datetime import datetime
from all_function import *
import pandas as pd
####关于数据库的信息配置
data_ip_set = '192.168.2.82'
data_port_set = 3306
data_user_set = 'root'
data_password_set = 'root'
data_name_set = 'szsw'

test_tab_root_name = 'tab_avg_root'
tab_train_12h_name = 'tab_train_12h'
tab_train_7d_name = 'tab_train_7d'
tab_predict_12h_temp_name = 'tab_predict_12h_temp'
tab_predict_7d_temp_name = 'tab_predict_7d_temp'
tab_predict_12h_record_name= 'tab_predict_12h_record'
tab_predict_7d_record_name= 'tab_predict_7d_record'

tab_create_key_name_arr = ['SnapTime','Chaetognatha','Medusae','Echinodermata','Shrimp','Copepoda','Appendicularia','Noctiluca','Temperature','Salinity']


####需要预测的生物种类\预测协变量种类，例如温盐深
need_predict_bio_arr = ['Medusae','Copepoda']
concomitant_variable_bio_arr = ['Temperature','Salinity']


####如果三个数据表不存在，建立数据表，如果存在，忽略
create_mysql(tab_create_key_name_arr,data_ip_set,data_port_set,data_user_set,data_password_set,data_name_set,tab_train_12h_name)
create_mysql(tab_create_key_name_arr,data_ip_set,data_port_set,data_user_set,data_password_set,data_name_set,tab_train_7d_name)
create_mysql(tab_create_key_name_arr,data_ip_set,data_port_set,data_user_set,data_password_set,data_name_set,tab_predict_12h_temp_name)
create_mysql(tab_create_key_name_arr,data_ip_set,data_port_set,data_user_set,data_password_set,data_name_set,tab_predict_7d_temp_name)
create_mysql(tab_create_key_name_arr,data_ip_set,data_port_set,data_user_set,data_password_set,data_name_set,tab_predict_12h_record_name)
create_mysql(tab_create_key_name_arr,data_ip_set,data_port_set,data_user_set,data_password_set,data_name_set,tab_predict_7d_record_name)
datebase_conn = mysqlConnect(data_ip_set,data_port_set,data_user_set,data_password_set,data_name_set)




tab_biology_avg_name = 'tab_biology_avg'


test_time_arr = ['2024-06-15 23:00:00']
before_12h_pred_1h_model_trian_time = datetime.now()


#####一天进行一次模型训练
next_train_12h_model_time = datetime.strptime('2024-06-15 20:00:00', "%Y-%m-%d %H:%M:%S")
# model_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'saved_models/nbeats_model.pkl')
# model_create_time = datetime.fromtimestamp(os.path.getatime(model_path))
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@',model_path,model_create_time)



######首先，每一小时驱动程序运行一次程序(测试改成人工输入时间)
for test_time_i in test_time_arr:
    try:
        #current_time = datetime.now()
        # if current_time.minute<5:
        #     ###运行一次程序，同时休眠半小时(后续所有功能全部写入此处)
        #     time.sleep(60*30)


        ###测试使用
        input_str_time = test_time_i
        current_time = datetime.strptime(input_str_time, "%Y-%m-%d %H:%M:%S")
        print(type(current_time))


        ##获取当前时间的前12个小时时间
        current_time_before_12h_arr = get_currenttime_before_12hour_fun(current_time)

        ###根据时间查询雷工的数据库，返回查询结果，如果没有进行补零在传回来
        return_all_info_list = find_mysql_match_12h_data(current_time_before_12h_arr,datebase_conn,tab_biology_avg_name,need_predict_bio_arr,concomitant_variable_bio_arr)

        temp_predict_value_list = {}
        temp_predict_value_list['SnapTime']=current_time_before_12h_arr[0]+timedelta(hours=1)
        print('预测的时间为：',temp_predict_value_list)
        #####根据传回来的数据字典送入预测网络进行预测
        for bio_i in return_all_info_list.keys():
            predict_12h_data = return_all_info_list[bio_i]

            # 获取预测值
            predictions = h12_predict_next_1h_points(predict_12h_data,bio_i,model_dir='saved_models')
            print(f"\n预测的未来1个时间点{current_time_before_12h_arr[0]+timedelta(hours=1)}物种{bio_i}的类数量:")
            predict_value = round(predictions[-1],2) if predictions[-1]>=0 else 0

            #print(f"预测值: {predictions[-1]:.2f}")
            print('预测值:',predict_value)
            temp_predict_value_list[bio_i] = predict_value

        #####进行数据库插入动作
        mysqlinsert_12h_predict(datebase_conn,tab_predict_12h_record_name,temp_predict_value_list.values(),temp_predict_value_list.keys())



        #####将当前时间的各种生物均值数据全部插入训练表，同时判断当前时间与表中上一条数据的时间间隔（如果间隔不是1h去雷工表中查询，没有补零），如果没有条数直接插入，最后读取表中所有数据返回
        write_h_data_to_train_tab(current_time_before_12h_arr[0],return_all_info_list,datebase_conn,tab_train_12h_name,need_predict_bio_arr,concomitant_variable_bio_arr,tab_biology_avg_name)


        #####加载训练表格进行模型训练,一天训练一次模型
        if (current_time - next_train_12h_model_time).days>0:
            read_12h_train_tab_to_train_model_fun(datebase_conn,tab_train_12h_name,need_predict_bio_arr)



        '''
            当前进度，已经写好12h预测未来1h值的基本代码，当前不确定问题，数据库没有温度、盐度相关信息
            下一进度：
            忘记写如果已经在数据库，就不要再插入数据库了
            7天预测1天数据
            12h预测模型训练（按天、按物种训练）
            7d预测模型训练（按天、按物种训练）
        '''
    except:
        print(traceback.print_exc())












#
#
#         ####按天训练预测模型，当模型的创建时间与当前时间存在一天差后进行模型训练
#         # current_time = datetime.now()
#         print('@@@@@@@@@@',(current_time -model_create_time).days)
#         if abs((current_time -model_create_time).days) > 0:
#             ###此处可以进行模型训练了
#             pass
#
#         #####模型训练，读取整个训练数据库
#         df = pd.read_sql("SELECT * FROM "+tab_train_12h_name, datebase_conn)
#         ###表头列名称
#         tab_train_12h_bio_names = df.columns.tolist()
#         tab_train_12h_row = df.shape[0]
#
#
#         #####添加数据，训练表最后一行数据与当前时间之间的数据查询后，全部添加进入train表（特殊，最开始表格为0只需要添加当前时刻值）
#         if tab_train_12h_row ==0:
#             ####添加当前时刻的数据值即可
#             pass
#         else:
#             #####添加数据表中的最后一行数据与当前时刻之间的数据值（补全时刻后，去雷工数据查询，没有置零-注意这里也可以先去雷工数据库查询，然后再验证时间的完整性）
#             tab_train_12h_last_row_data = df.iloc[-1]
#             #print('tab_train_12h_last_row_data',tab_train_12h_last_row_data)
#
#
#         #####训练表格
#         if tab_train_12h_row <24:
#             ###跳过不训练
#             pass
#         else:
#             ####训练之前要保证数据的连续性，至少一天内的数据要一小时一个点
#
#             ####模型数据大于24行，可以进行训练
#             pass
#
#
#
#
#
#
#         #####模型训练好之后，取一定的值进行预测
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#         next_iter_input = input('是否进行下一次计算，任意输入即可:')
#     except:
#         print(traceback.print_exc())
#
#
#
#
# ###查询训练表中数据，如果没有，从现在开始添加，如果有取到最大时间，同时取出所有数据，判断是否大于12小时，是否大于7天数据
# #获取预测数据表中最近时间
# query_latest_data_sql = f"SELECT * FROM {tab_train_name} WHERE SnapTime = (SELECT MAX(SnapTime) FROM {tab_train_name});"
# database_cursor.execute(query_latest_data_sql)
# latest_data_info = database_cursor.fetchone()
#
# ##如果最近一笔时间没有，则从这里开始添加数据
# if latest_data_info==None:
#     print('fsfsdf',latest_data_info)
#
# ###如果有数据，获取最近数据，然后去取雷工的均值表填入，再获取表中所有数据进行训练，获取最近最佳预测数据预测
# else:
#     print(latest_data_info)
#
#
# ###每小时读取一次均值表值，测试估计需要想其它方案
# ####第一次预测，什么表都没有
# ####第n次进行预测，表中有部分数据
"""
