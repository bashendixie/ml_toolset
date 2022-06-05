import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from cycler import cycler


def aaa():
    oldcycler = plt.rcParams['axes.prop_cycle']
    plt.rcParams['axes.facecolor'] = '#0057b8' # blue
    plt.rcParams['axes.prop_cycle'] = cycler(color=['#ffd700'] +
                                             oldcycler.by_key()['color'][1:])

    # Read the data
    train = pd.read_csv('origin/train.csv', parse_dates=['time'])
    test = pd.read_csv('origin/test.csv', index_col='row_id', parse_dates=['time'])

    # Feature Engineering
    for df in [train, test]:
        df['day'] = df.time.dt.day
        df['weekday'] = df.time.dt.weekday
        df['hour'] = df.time.dt.hour
        df['minute'] = df.time.dt.minute

    filter1 = train['x']=='0'
    filter2 = train['y']=='0'
    filter3 = train['direction'] =='0'
    filter4 = train['minute']=='0'
    train.where(filter1 & filter2 & filter3 & filter4)
    train.to_csv('group_and_sums_1.csv', index=False)
    #medians = train.groupby(['x', 'y', 'direction', 'day', 'hour', 'minute']).congestion.sum().astype(int)
    #medians.to_csv('group_and_sums_1.csv', index=False)

    # Compute the median congestion for every place and time of week
    #sums = train.groupby(['hour', 'minute']).congestion.sum().astype(int)
    #print(sums)

#sums.to_csv('weekday_group_and_sums_1.csv', index=False)


def bbb():
    train = pd.read_csv('train_action_del_half.csv', parse_dates=['time'])
    train['hour'] = train['time'].dt.hour
    train.drop(train[train['hour'] < 12].index)
    train.to_csv('del.csv')

bbb()