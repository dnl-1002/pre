
"""
12h -> 1h plankton abundance prediction entrypoint.

This script is intended to be called once per hour by an external scheduler.
It reads the latest 12 hourly records, predicts the next hour for each target
species, writes one complete prediction row, then optionally retrains the
daily models after the prediction succeeds.

The legacy exploratory code is kept below for reference, but normal
`python start_main.py` execution exits through this entrypoint before reaching it.
"""

import os as _os
import sys as _sys
import traceback as _traceback
from datetime import datetime as _datetime, timedelta as _timedelta

from all_function import (
    find_mysql_match_12h_data as _find_mysql_match_12h_data,
    get_currenttime_before_12hour_fun as _get_currenttime_before_12hour_fun,
    write_h_data_to_train_tab as _write_h_data_to_train_tab,
)
from h12_model_train import read_12h_train_tab_to_train_model_fun as _read_12h_train_tab_to_train_model_fun
from nbeats.nbeats_model import h12_predict_next_1h_points as _h12_predict_next_1h_points
from pymysql_data import create_mysql as _create_mysql
from pymysql_data import mysqlConnect as _mysqlConnect
from pymysql_data import mysqlinsert_12h_predict as _mysqlinsert_12h_predict


_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_os.chdir(_SCRIPT_DIR)


#### Database configuration.
DATA_IP = "192.168.2.82"
DATA_PORT = 3306
DATA_USER = "root"
DATA_PASSWORD = "root"
DATA_NAME = "szsw"


#### Table configuration used by the first stable version.
SOURCE_AVG_TABLE = "tab_biology_avg"
TRAIN_12H_TABLE = "tab_train_12h"
PREDICT_12H_RECORD_TABLE = "tab_predict_12h_record"


#### Species and exogenous variables.
NEED_PREDICT_BIO = ["Medusae", "Copepoda"]
CONCOMITANT_VARIABLES = ["Temperature", "Salinity"]


#### Model and runtime configuration.
MODEL_DIR = "saved_models"
ENABLE_TEST_TIME_OVERRIDE = False
TEST_CURRENT_TIME = "2024-06-15 23:00:00"


#### Fields used when creating the training and prediction tables.
TABLE_CREATE_COLUMNS = [
    "SnapTime",
    "Chaetognatha",
    "Medusae",
    "Echinodermata",
    "Shrimp",
    "Copepoda",
    "Appendicularia",
    "Noctiluca",
    "Temperature",
    "Salinity",
]


def _get_current_hour():
    """Return the scheduler time rounded down to the hour."""
    if ENABLE_TEST_TIME_OVERRIDE:
        return _datetime.strptime(TEST_CURRENT_TIME, "%Y-%m-%d %H:%M:%S")
    return _datetime.now().replace(minute=0, second=0, microsecond=0)


def _ensure_required_tables():
    """Create only the first-version tables required by the 12h pipeline."""
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


def _connect_database():
    return _mysqlConnect(DATA_IP, DATA_PORT, DATA_USER, DATA_PASSWORD, DATA_NAME)


def _validate_model_files():
    """Fail early if any required species model is missing."""
    missing_models = []
    for bio_name in NEED_PREDICT_BIO:
        model_path = _os.path.join(MODEL_DIR, f"{bio_name}_nbeats_model.pkl")
        if not _os.path.exists(model_path):
            missing_models.append(model_path)
    if missing_models:
        raise FileNotFoundError("缺少模型文件: " + ", ".join(missing_models))


def _should_retrain_today(current_time):
    """Retrain when any target species model file is older than current date."""
    current_date = current_time.date()
    for bio_name in NEED_PREDICT_BIO:
        model_path = _os.path.join(MODEL_DIR, f"{bio_name}_nbeats_model.pkl")
        if not _os.path.exists(model_path):
            return True
        model_date = _datetime.fromtimestamp(_os.path.getmtime(model_path)).date()
        if model_date < current_date:
            print(f"{bio_name} 模型日期为 {model_date}，早于当前日期 {current_date}，需要重训")
            return True
    return False


def _predict_all_species(history_by_species, target_time):
    """Predict all species first, then return one complete DB row."""
    predict_row = {"SnapTime": target_time}

    for bio_name in NEED_PREDICT_BIO:
        if bio_name not in history_by_species:
            raise KeyError(f"查询结果中缺少物种 {bio_name}")

        historical_data = history_by_species[bio_name]
        predictions = _h12_predict_next_1h_points(historical_data, bio_name, model_dir=MODEL_DIR)
        if len(predictions) == 0:
            raise ValueError(f"{bio_name} 模型未返回预测值")

        raw_value = float(predictions[-1])
        predict_value = round(raw_value, 2) if raw_value >= 0 else 0
        predict_row[bio_name] = predict_value
        print(f"预测时间 {target_time}，物种 {bio_name}，预测值 {predict_value}")

    return predict_row


def _run_once():
    current_time = _get_current_hour()
    target_time = current_time + _timedelta(hours=1)
    print(f"当前整点: {current_time}")
    print(f"预测目标时间: {target_time}")

    _ensure_required_tables()
    conn = _connect_database()
    try:
        _validate_model_files()

        before_12h_times = _get_currenttime_before_12hour_fun(current_time)
        history_by_species = _find_mysql_match_12h_data(
            before_12h_times,
            conn,
            SOURCE_AVG_TABLE,
            NEED_PREDICT_BIO,
            CONCOMITANT_VARIABLES,
        )

        predict_row = _predict_all_species(history_by_species, target_time)

        # Write only after every target species has produced a valid prediction.
        _mysqlinsert_12h_predict(
            conn,
            PREDICT_12H_RECORD_TABLE,
            predict_row.values(),
            predict_row.keys(),
        )

        _write_h_data_to_train_tab(
            current_time,
            history_by_species,
            conn,
            TRAIN_12H_TABLE,
            NEED_PREDICT_BIO,
            CONCOMITANT_VARIABLES,
            SOURCE_AVG_TABLE,
        )

        if _should_retrain_today(current_time):
            print("开始每日模型重训")
            _read_12h_train_tab_to_train_model_fun(conn, TRAIN_12H_TABLE, NEED_PREDICT_BIO)
        else:
            print("今日模型已训练，无需重训")
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
Legacy exploratory code retained for reference only.

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
