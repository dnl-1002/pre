import time,pymysql
from datetime import datetime,timedelta
import pandas as pd

###尝试连接数据库
def mysqlConnect(data_ip_set,data_port_set,data_user_set,data_password_set,data_name_set):
    conn = pymysql.connect(host=data_ip_set, port=data_port_set, user=data_user_set, passwd=data_password_set, db=data_name_set,charset='utf8')
    return conn




#####查询多天数据
def continuous_time_extend_fun(start_time,over_time):
    start_mid_over_time_arr = []

    ###头部时间添加
    print(start_time.hour)
    for i_t in range(int(start_time.hour)+1,24):
        if i_t < 10:
            i_t = '0' + str(i_t)
        t_temp_str_time = str(i_t)+':00:00'
        t_temp_data_str_time = str(start_time.date()) + ' ' + t_temp_str_time
        t_temp_data_time = datetime.strptime(t_temp_data_str_time, "%Y-%m-%d %H:%M:%S")
        start_mid_over_time_arr.append(t_temp_data_time)

    ####去头，去尾时间循环添加
    temp_data = start_time.date()
    while temp_data < over_time.date()-timedelta(days=1):
        temp_data += timedelta(days=1)
        for i in range(0,24):
            if i < 10:
                i = '0'+str(i)
            temp_str_time = str(i)+':00:00'
            temp_data_str_time = str(temp_data)+' '+temp_str_time
            temp_data_time = datetime.strptime(temp_data_str_time, "%Y-%m-%d %H:%M:%S")
            start_mid_over_time_arr.append(temp_data_time)

    #####尾部时间添加
    for i_w in range(0,int(over_time.hour)):
        if i_w < 10:
            i_w = '0' + str(i_w)
        w_temp_str_time = str(i_w) + ':00:00'
        w_temp_data_str_time = str(over_time.date()) + ' ' + w_temp_str_time
        w_temp_data_time = datetime.strptime(w_temp_data_str_time, "%Y-%m-%d %H:%M:%S")
        start_mid_over_time_arr.append(w_temp_data_time)
    #print(start_mid_over_time_arr)
    # for i in start_mid_over_time_arr:
    #     print(i)
    return start_mid_over_time_arr


##########查询两个时间之间的时间，按小时为间隔
def start_over_time_genorate_mid_time(start_time,over_time):
    start_mid_over_time_arr = []
    temp_data = start_time
    while temp_data < over_time+timedelta(hours=-1):
        temp_data += timedelta(hours=1)
        print(temp_data)
        start_mid_over_time_arr.append(temp_data)
    print(start_mid_over_time_arr)
    return  start_mid_over_time_arr





###########将结果插入12h小时训练表中(中间没有缺失数据)
def insert_12h_train_tab_fun(current_predict_time,current_time_all_info_list,need_predict_bio_arr,concomitant_variable_bio_arr,tab_train_12h_name,datebase_cursor,datebase_conn):
    columns_arr, data_arr = [], []

    ####添加时间
    columns_arr.append('Snaptime')
    data_arr.append(current_predict_time)

    ##添加生物数量信息
    for signal_bio_key in current_time_all_info_list.keys():
        if signal_bio_key in need_predict_bio_arr:
            columns_arr.append(signal_bio_key)
            data_arr.append(current_time_all_info_list[signal_bio_key]['y'][-1])

    #####添加温盐信息
    for concomitant_variable_i in current_time_all_info_list[need_predict_bio_arr[0]].keys():
        if concomitant_variable_i in concomitant_variable_bio_arr:
            columns_arr.append(concomitant_variable_i)
            data_arr.append(current_time_all_info_list[need_predict_bio_arr[0]][concomitant_variable_i][-1])

    ####转换成元组
    values_arr = ['%s'] * len(columns_arr)
    columns = ', '.join([f"{keyword} " for keyword in columns_arr])
    values_num = ', '.join([f"{keyword} " for keyword in values_arr])
    tuple_data = tuple(data_arr)

    #insert_query = f"INSERT INTO {tab_train_12h_name} ({columns}) VALUES ({values_num})"
    #datebase_cursor.execute(insert_query, tuple_data)
    ####解决重复插入问题
    insert_query = f"""INSERT INTO {tab_train_12h_name} ({columns}) SELECT {values_num} FROM DUAL WHERE NOT EXISTS(SELECT 1 FROM {tab_train_12h_name} WHERE Snaptime = %s)"""
    datebase_cursor.execute(insert_query, tuple_data + (tuple_data[0],))
    datebase_conn.commit()

###########将结果插入12h小时训练表中(中间缺失数据)
def insert_12h_train_tab_loss_data_fun(start_over_time_mid_time_arr,find_loss_data_list,need_predict_bio_arr,concomitant_variable_bio_arr,tab_train_12h_name,datebase_cursor,datebase_conn):
    #start_over_time_mid_time_arr, find_loss_data_list
    for index in range(len(start_over_time_mid_time_arr)):
        columns_arr, data_arr = [], []

        ####添加时间
        columns_arr.append('Snaptime')
        data_arr.append(start_over_time_mid_time_arr[index])

        ##添加生物数量信息
        for signal_bio_key in find_loss_data_list.keys():
            if signal_bio_key in need_predict_bio_arr:
                columns_arr.append(signal_bio_key)
                data_arr.append(find_loss_data_list[signal_bio_key]['y'][index])

        #####添加温盐信息
        for concomitant_variable_i in find_loss_data_list[need_predict_bio_arr[0]].keys():
            if concomitant_variable_i in concomitant_variable_bio_arr:
                columns_arr.append(concomitant_variable_i)
                data_arr.append(find_loss_data_list[need_predict_bio_arr[0]][concomitant_variable_i][index])

        ####转换成元组
        values_arr = ['%s'] * len(columns_arr)
        columns = ', '.join([f"{keyword} " for keyword in columns_arr])
        values_num = ', '.join([f"{keyword} " for keyword in values_arr])
        tuple_data = tuple(data_arr)

        # insert_query = f"INSERT INTO {tab_train_12h_name} ({columns}) VALUES ({values_num})"
        # datebase_cursor.execute(insert_query, tuple_data)
        insert_query = f"""INSERT INTO {tab_train_12h_name} ({columns}) SELECT {values_num} FROM DUAL WHERE NOT EXISTS(SELECT 1 FROM {tab_train_12h_name} WHERE Snaptime = %s)"""
        datebase_cursor.execute(insert_query, tuple_data + (tuple_data[0],))
    datebase_conn.commit()





#####查询训练数据库中缺失的数据值
def find_many_point_time_tab_fun(final_data_snaptime,current_predict_time,start_over_time_mid_time_arr, datebase_conn,tab_biology_avg_name, need_predict_bio_arr,concomitant_variable_bio_arr):
    # 构造 SQL 查询语句
    find_sql = f"""SELECT * FROM {tab_biology_avg_name} WHERE SnapTime > '{str(final_data_snaptime)}' 
       AND SnapTime < '{str(current_predict_time)}' AND SummaryInterval=60"""
    print(find_sql)

    # 使用 pandas 读取数据
    df = pd.read_sql(find_sql, datebase_conn)
    all_time_arr = df['SnapTime'].tolist()
    ###定义返回结果
    return_all_info_list = {}

    ####物种丰度信息获取
    for bio_i in need_predict_bio_arr:
        #####生物丰度值查询与补充
        temp_signal_bio_list = {}
        temp_bio_i_fd_arr = []
        for time_i in start_over_time_mid_time_arr:
            if time_i in all_time_arr:
                try:
                    row_index = df.isin([time_i]).any(axis=1)  # 判断哪些行包含目标值
                    # print('单一值',time_i,bio_i,float(df.loc[row_index,bio_i]))
                    temp_bio_i_fd_arr.append(float(df.loc[row_index, bio_i]))
                except:
                    ###如果报错，说明查询的物种不存在
                    break
            else:
                ###如果部分时间数据没有，直接补零
                temp_bio_i_fd_arr.append(0.0)
                # print('@@@@@@@@@@@@@22',time_i)
        temp_bio_i_fd_arr_reverse = list(reversed(temp_bio_i_fd_arr))
        temp_signal_bio_list['y'] = temp_bio_i_fd_arr_reverse
        return_all_info_list[bio_i] = temp_signal_bio_list

        ####协变量数组定义查询与补零
        for num, concomitant_variable_i in enumerate(concomitant_variable_bio_arr):
            temp_concomitant_variable_arr = []
            for time_i in start_over_time_mid_time_arr:
                if time_i in all_time_arr:
                    try:
                        row_index = df.isin([time_i]).any(axis=1)  # 判断哪些行包含目标值
                        # print('协变量', time_i, bio_i, float(df.loc[row_index, concomitant_variable_i]))
                        temp_concomitant_variable_arr.append(float(df.loc[row_index, concomitant_variable_i]))
                    except:
                        ###如果协变量查询报错，说明不存在协变量关键字
                        temp_concomitant_variable_arr.append(0.0)
                else:
                    ###如果协变量部分时间不存在，则取插值
                    temp_concomitant_variable_arr.append(0.0)
            temp_concomitant_variable_arr_reverse = list(reversed(temp_concomitant_variable_arr))
            temp_signal_bio_list[concomitant_variable_i] = temp_concomitant_variable_arr_reverse

    print(return_all_info_list)
    return return_all_info_list




#####将当前时间的各种生物均值数据全部插入训练表，同时判断当前时间与表中上一条数据的时间间隔（如果间隔不是1h去雷工表中查询，没有补零），如果没有条数直接插入，最后读取表中所有数据返回
def write_h_data_to_train_tab(current_predict_time,current_time_all_info_list,datebase_conn,tab_train_12h_name,need_predict_bio_arr,concomitant_variable_bio_arr,tab_biology_avg_name):

    # 构造 SQL 查询语句,查询数据库最后一条数据
    sql = f"""SELECT Snaptime FROM {tab_train_12h_name} ORDER BY id DESC LIMIT 1"""
    print(sql)
    datebase_cursor = datebase_conn.cursor()
    datebase_cursor.execute(sql)

    # 获取结果
    result = datebase_cursor.fetchone()  # fetchone() 返回一个字典
    if result:
        ###如果有最后一条数据值，看当前时间值与最后一条值是否间隔一小时，不是去雷工数据查询，是直接插入数据库
        try:
            final_data_snaptime = datetime.strptime(result[0], "%Y/%m/%d %H:%M:%S")
        except:
            final_data_snaptime = datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S")
        print('@@@@@@@@final_data_snaptime',final_data_snaptime,current_predict_time,type(final_data_snaptime),type(current_predict_time))
        start_over_time_mid_time_arr = start_over_time_genorate_mid_time(final_data_snaptime,current_predict_time)
        if len(start_over_time_mid_time_arr)>0:
            ####中间有缺失的数据去雷工数据库查询
            find_loss_data_list = find_many_point_time_tab_fun(final_data_snaptime,current_predict_time,start_over_time_mid_time_arr, datebase_conn,tab_biology_avg_name, need_predict_bio_arr,concomitant_variable_bio_arr)
            print('@@@@@@@@111find_loss_data_list',find_loss_data_list)
            insert_12h_train_tab_loss_data_fun(start_over_time_mid_time_arr, find_loss_data_list, need_predict_bio_arr,concomitant_variable_bio_arr, tab_train_12h_name, datebase_cursor, datebase_conn)
            ###插入完成缺失的值后还需要插入一遍当前值
            insert_12h_train_tab_fun(current_predict_time, current_time_all_info_list, need_predict_bio_arr, concomitant_variable_bio_arr, tab_train_12h_name, datebase_cursor, datebase_conn)

        else:
            ######直接插入数据库，和下面初始数据库没有值，相同
            insert_12h_train_tab_fun(current_predict_time, current_time_all_info_list, need_predict_bio_arr,concomitant_variable_bio_arr, tab_train_12h_name, datebase_cursor, datebase_conn)
    else:
        ####没有数据直接插入当前数据进入数据库
        insert_12h_train_tab_fun(current_predict_time,current_time_all_info_list,need_predict_bio_arr,concomitant_variable_bio_arr,tab_train_12h_name,datebase_cursor,datebase_conn)










#####获取当前时间的前12小时数组
def get_currenttime_before_12hour_fun(currenttime):
    return_12hour_time_arr = []
    for i in range(0,12):
        temp_time_hour = currenttime - timedelta(hours=i)
        temp_time_date = temp_time_hour.date()
        temp_time_time_h = temp_time_hour.hour
        if int(temp_time_time_h)<10:
            temp_time_time_h = '0'+str(temp_time_time_h)
        temp_time_time = str(temp_time_time_h)+':00:00'
        temp_time_combine = str(temp_time_date) + ' ' +temp_time_time
        temp_time_append= datetime.strptime(temp_time_combine, "%Y-%m-%d %H:%M:%S")
        print(temp_time_date,temp_time_time)
        return_12hour_time_arr.append(temp_time_append)
    return return_12hour_time_arr

####根据传入的时间数组去查询数据库，获取时间点对应的数据；缺失时间点使用线性插值和前后填充，最后兜底补零
def find_mysql_match_12h_data(current_time_before_12h_arr,datebase_conn,table_name,need_predict_bio_arr,concomitant_variable_bio_arr):
    '''
    返回目标值：
     example_data = {生物名称：
     {
        'y': [0.178571429, 0.465116279, 0.315315315, 0.430463576, 0.616883117, 0.460122699, 0.728476821, 1.006289308, 1.231884058, 0.172413793, 0, 0.151515152],  # 桡足类数量
        'Temp_C': [19.64, 19.7525, 19.835, 19.8825, 19.9275, 19.915, 20.0525, 20.315, 20.405, 20.445, 20.13, 19.995],  # 温度
        'Sal': [33.095, 33.1275, 33.1725, 33.1675, 33.125, 33.0625, 33.0625, 33.13, 33.13, 33.03, 32.965, 33.01]  # 盐度
    }
    }
    '''

    # 构造 SQL 查询语句
    find_sql = f"""SELECT * FROM {table_name} WHERE SnapTime >= '{current_time_before_12h_arr[-1]}' 
    AND SnapTime <= '{current_time_before_12h_arr[0]}' AND SummaryInterval=60 AND DeviceID=1"""  

    # 使用 pandas 读取数据
    df = pd.read_sql(find_sql, datebase_conn)
    target_times = sorted(pd.to_datetime(current_time_before_12h_arr))
    target_index = pd.DatetimeIndex(target_times)
    value_columns = list(dict.fromkeys(need_predict_bio_arr + concomitant_variable_bio_arr))

    if 'SnapTime' not in df.columns:
        df = pd.DataFrame(columns=['SnapTime'] + value_columns)
    df['SnapTime'] = pd.to_datetime(df['SnapTime'], errors='coerce')
    df = df.dropna(subset=['SnapTime']).sort_values('SnapTime')

    for column_name in value_columns:
        if column_name not in df.columns:
            df[column_name] = 0.0

    if df.empty:
        window_df = pd.DataFrame(index=target_index, columns=value_columns)
    else:
        window_df = df[['SnapTime'] + value_columns].copy()
        window_df = window_df.drop_duplicates(subset=['SnapTime'], keep='last')
        window_df = window_df.set_index('SnapTime').reindex(target_index)

    for column_name in value_columns:
        window_df[column_name] = pd.to_numeric(window_df[column_name], errors='coerce')
    window_df = window_df.interpolate(method='linear', limit_direction='both').ffill().bfill().fillna(0.0)

    ###定义返回结果，顺序为从旧到新，保持模型输入与训练时间方向一致
    return_all_info_list = {}
    for bio_i in need_predict_bio_arr:
        temp_signal_bio_list = {'y': window_df[bio_i].astype(float).tolist()}
        for concomitant_variable_i in concomitant_variable_bio_arr:
            temp_signal_bio_list[concomitant_variable_i] = window_df[concomitant_variable_i].astype(float).tolist()
        return_all_info_list[bio_i] = temp_signal_bio_list

    print(return_all_info_list)
    return return_all_info_list





if __name__ == '__main__':
    # start_time = '2024/03/20 00:00'
    # over_time = '2024/03/21 23:00'
    # datatime_start_time = datetime.strptime(start_time, "%Y/%m/%d %H:%M")
    # datatime_over_time = datetime.strptime(over_time, "%Y/%m/%d %H:%M")
    # start_over_time_genorate_mid_time(datatime_start_time,datatime_over_time)
    # start_time = '202/10/25 00:00'
    # print(get_currenttime_before_12hour_fun(datetime.now()))
    pass


    ##############这里要编写一个批量获取之前数据的脚本
    write_h_data_to_train_tab(current_time_before_12h_arr[0], return_all_info_list, datebase_conn, tab_train_12h_name,
                              need_predict_bio_arr, concomitant_variable_bio_arr, tab_biology_avg_name)

