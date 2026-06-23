
import time,os
from datetime import datetime,timedelta

# input_start_time = input('开始日期yyyy/mm/dd：')
# input_over_time =  input('结束日期yyyy/mm/dd：')
# start_time = datetime.strptime(input_start_time, "%Y/%m/%d")
# over_time = datetime.strptime(input_over_time, "%Y/%m/%d")
#
# start_over_time_arr = []
#
# current_date = start_time
# start_over_time_arr.append(str(start_time.date()))
# while current_date < over_time:
#     current_date += timedelta(days=1)
#     start_over_time_arr.append(str(current_date.date()))
# print(start_over_time_arr)

model_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'saved_models/nbeats_model.pkl')
model_create_time = datetime.fromtimestamp(os.path.getatime(model_path))
current_time = datetime.now()
print(type(model_create_time),type(current_time))
print(current_time)
print((current_time-model_create_time).days)
print((model_create_time-current_time).days)