import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import borax
import time,datetime
from borax.calendars import LunarDate
from borax.calendars.festivals import get_festival, LunarSchema, DayLunarSchema

datas = pd.read_csv('test_aaa.csv')
one_hot_data = pd.get_dummies(datas, prefix=['country', 'store', 'product'])
print(one_hot_data.head())
one_hot_data.to_csv("test_bbb.csv")


def str2date(str, date_format="%Y/%m/%d"):
  date = datetime.datetime.strptime(str, date_format)
  return date

def get_weekday(date):
  week_day_dict = {
    0 : 0,
    1 : 1,
    2 : 2,
    3 : 3,
    4 : 4,
    5 : 5,
    6 : 6,
  }
  day = str2date(date).weekday()
  return week_day_dict[day]



def get_delta_days(t1, t2):
  dt2 = datetime.datetime.fromtimestamp(t2)
  dt2 = dt2.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
  dt1 = datetime.datetime.fromtimestamp(t1)
  dt1 = dt1.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
  return (dt2 - dt1).days


# 读取数据
datas = pd.read_csv('test_origin.csv')

# 挖掘，周几对交易量的影响
datas['Weekday'] = datas['date']
datas['Weekday'] = datas['Weekday'].map(lambda x: get_weekday(x))

# 挖掘，各个节日对交易量的影响
datas['Festival'] = datas['date']
datas['Festival'] = datas['Festival'].map(lambda x: get_festival(x))

# 将 Date 一列变为与 2016-10-30 这天的距离
datas['date'] = datas['date'].map(lambda x: get_delta_days(str2date(x).timestamp(), str2date('2015/1/1').timestamp()))

# 缩小数据大小
datas['num_sold'] = datas['num_sold'].map(lambda x: (float(x) / 1000.))

print(datas.head())
datas.to_csv("test_aaa.csv")

