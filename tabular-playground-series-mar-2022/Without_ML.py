import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from cycler import cycler
from IPython import display

oldcycler = plt.rcParams['axes.prop_cycle']
plt.rcParams['axes.facecolor'] = '#0057b8' # blue
plt.rcParams['axes.prop_cycle'] = cycler(color=['#ffd700'] +
                                         oldcycler.by_key()['color'][1:])

def calcMean():
    # Read the data
    train = pd.read_csv('origin/train.csv', index_col='row_id', parse_dates=['time'])
    test = pd.read_csv('origin/test.csv', index_col='row_id', parse_dates=['time'])

    # Feature Engineering
    for df in [train, test]:
        df['weekday'] = df.time.dt.weekday
        df['hour'] = df.time.dt.hour
        df['minute'] = df.time.dt.minute

    # Compute the median congestion for every place and time of week
    medians = train.groupby(['x', 'y', 'direction', 'weekday', 'hour', 'minute']).congestion.median().astype(int)
    print(medians)

    means = train.groupby(['x', 'y', 'direction', 'weekday', 'hour', 'minute']).congestion.mean().astype(int)
    print(means)

    maxs = train.groupby(['x', 'y', 'direction', 'weekday', 'hour', 'minute']).congestion.max().astype(int)
    print(maxs)

    mins = train.groupby(['x', 'y', 'direction', 'weekday', 'hour', 'minute']).congestion.min().astype(int)
    print(mins)

    halfs = (medians + means) / 2
    print(halfs)

    threes = (medians + means + (maxs-mins)) / 3
    print(threes)

    # Write the submission file
    sub = test.merge(threes,
                     left_on=['x', 'y', 'direction', 'weekday', 'hour', 'minute'],
                     right_index=True)[['congestion']]
    sub.reset_index(inplace=True)
    sub.to_csv('submission_no_machine_learning_v1.csv', index=False)
    print(sub.head())

def calcQuantile():
    train = pd.read_csv('origin/train.csv', parse_dates=['time'])
    test = pd.read_csv('origin/test.csv', index_col='row_id', parse_dates=['time'])

    for df in [train, test]:
        df['weekday'] = df.time.dt.weekday
        df['hour'] = df.time.dt.hour
        df['minute'] = df.time.dt.minute
        df['key'] = str(df['hour']) + str(df['minute'])+ str(df['x'])+ str(df['y']) + str(df['direction'])


    # 第一步，从训练数据获取部分，然后在获取平均值
    # Compute the quantiles of workday afternoons in September except Labor Day
    sep = train[(train.time.dt.hour >= 12) & (train.time.dt.weekday < 5) & (train.time.dt.dayofyear >= 246)]
    lower = sep.groupby(['key']).congestion.quantile(0.15).values
    upper = sep.groupby(['key']).congestion.quantile(0.85).values

    sums = sep.groupby(['key']).congestion.sum()

    # Clip the submission data to the quantiles
    train_out = train.copy()
    train_out['congestion'] = train_out.congestion.clip(lower, upper)

    # Compute the median congestion for every place and time of week
    medians = train_out.groupby(['x', 'y', 'direction',  'hour', 'minute']).congestion.median().astype(int)
    print(medians)

    means = train_out.groupby(['x', 'y', 'direction', 'hour', 'minute']).congestion.mean().astype(int)
    print(means)

    halfs = (medians + means) / 2
    print(halfs)

    # Write the submission file
    sub = test.merge(halfs,
                     left_on=['x', 'y', 'direction', 'hour', 'minute'],
                     right_index=True)[['congestion']]
    sub.reset_index(inplace=True)
    sub.to_csv('submission_no_machine_learning_v2.csv', index=False)
    print(sub.head())


calcQuantile()


# # Plot the distribution of the test predictions
# # compared to the other Monday afternoons
# plt.figure(figsize=(16,3))
# plt.hist(train.congestion[((train.time.dt.weekday == 0) &
#                            (train.time.dt.hour >= 12)).values],
#          bins=np.linspace(-0.5, 100.5, 102),
#          density=True, label='Train',
#          color='#ffd700')
# plt.hist(sub['congestion'], np.linspace(-0.5, 100.5, 102),
#          density=True, rwidth=0.5, label='Test predictions',
#          color='r')
# plt.xlabel('Congestion')
# plt.ylabel('Frequency')
# plt.title('Congestion on Monday afternoons')
# plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
# plt.legend()
# plt.show()
