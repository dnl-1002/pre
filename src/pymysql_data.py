'''
数据库操作动作
'''
import pymysql


def mysql_column_type(column_name):
    """根据业务字段返回 MySQL 列类型。"""
    if column_name in ("SnapTime", "PredictSnapTime"):
        return "DATETIME"
    if column_name == "Id":
        return "INT"
    return "DOUBLE"



#####12h小时预测结果插入
def mysqlinsert_12h_predict(conn,tab_name,signal_info_arr,key_name_arr):
    # 数据库游标！
    cur = conn.cursor()
    values_arr = ['%s']* len(key_name_arr)
    columns = ', '.join([f"{keyword} " for keyword in key_name_arr])
    values_num = ', '.join([f"{keyword} " for keyword in values_arr])
    signal_info_tuple = tuple(signal_info_arr)


    ###对数据库进行数据插入，如果不用txt文本插入，可以直接在这个函数传入要插入的数据流
    try:
        # insert_query = f"INSERT INTO {tab_name} ({columns}) VALUES ({values_num})"
        # cur.execute(insert_query, signal_info_tuple)
        insert_query = f"""INSERT INTO {tab_name} ({columns}) SELECT {values_num} FROM DUAL WHERE NOT EXISTS(SELECT 1 FROM {tab_name} WHERE Snaptime = %s)"""
        cur.execute(insert_query, signal_info_tuple + (signal_info_tuple[0],))
    except Exception as e:
        conn.rollback()
        print('Insert error:', e)
        raise

    # 真正的执行语句
    cur.close()
    conn.commit()
    print("数据库插入完成")

def mysqlinsert(conn,tab_name,signal_info_arr,key_name_arr):
    # 数据库游标！
    cur = conn.cursor()
    values_arr = ['%s']* len(key_name_arr)
    columns = ', '.join([f"{keyword} " for keyword in key_name_arr])
    values_num = ', '.join([f"{keyword} " for keyword in values_arr])
    signal_info_tuple = tuple(signal_info_arr)


    ###对数据库进行数据插入，如果不用txt文本插入，可以直接在这个函数传入要插入的数据流
    try:
        insert_query = f"INSERT INTO {tab_name} ({columns}) VALUES ({values_num})"
        cur.execute(insert_query, signal_info_tuple)
    except Exception as e:
        print('Insert error:', e)

    # 真正的执行语句
    cur.close()
    conn.commit()
    print("数据库插入完成")



###尝试连接数据库
def mysqlConnect(data_ip_set,data_port_set,data_user_set,data_password_set,data_name_set):
    conn = pymysql.connect(host=data_ip_set, port=data_port_set, user=data_user_set, passwd=data_password_set, db=data_name_set,charset='utf8')
    return conn


#####第一次创建数据库
def create_mysql(key_name_arr,data_ip_set,data_port_set,data_user_set,data_password_set,data_name_set,tab_name):
    ###尝试连接数据库，进行数据库创建
    conn = pymysql.connect(host=data_ip_set, port=data_port_set, user=data_user_set, passwd=data_password_set,
                           charset='utf8')
    cur = conn.cursor()
    k = "create database if not exists " + data_name_set
    cur.execute(k)
    conn.commit()
    print("数据库已经建立")
    cur.close()
    conn.close()

    ###尝试连接数据库进行数据表创建
    conn = pymysql.connect(host=data_ip_set, port=data_port_set, user=data_user_set, passwd=data_password_set,
                           db=data_name_set, charset='utf8')
    cur = conn.cursor()
    columns = ', '.join([f"`{keyword}` {mysql_column_type(keyword)}" for keyword in key_name_arr])
    id_auto_add =  'Id INT AUTO_INCREMENT PRIMARY KEY,'
    print('@@@@@@',columns)
    create_table_query = f"CREATE TABLE IF NOT EXISTS {tab_name} ({id_auto_add+columns})"
    cur.execute(create_table_query)
    cur.close()
    conn.commit()
    print('数据表创建成功！')
