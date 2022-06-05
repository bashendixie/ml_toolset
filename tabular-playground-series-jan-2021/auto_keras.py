# use autokeras to find a model for the insurance dataset
from numpy import asarray
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from autokeras import StructuredDataRegressor
import tensorflow as tf
from keras.models import Sequential, load_model


train = pd.read_csv('group/merge.csv', index_col='id')
test = pd.read_csv('origin/test.csv', index_col='id')
ss = pd.read_csv('origin/sample_submission.csv')

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col!= 'time':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

y = train['target']
del train['target']
# del test['cont9']

train['max'] = train.max(axis=1)
train['min'] = train.min(axis=1)
train['mean'] = train.mean(axis=1)
train['sum'] = train.sum(axis=1)
train['cha'] = train.max(axis=1) - train.min(axis=1)
train['zhong'] = (train.max(axis=1) + train.min(axis=1))/2

test['max'] = test.max(axis=1)
test['min'] = test.min(axis=1)
test['mean'] = test.mean(axis=1)
test['sum'] = test.sum(axis=1)
test['cha'] = test.max(axis=1) - test.min(axis=1)
test['zhong'] = (test.max(axis=1) + test.min(axis=1))/2

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)



def train_1():
    # # load dataset
    # url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
    # dataframe = read_csv(url, header=None)
    # print(dataframe.shape)
    # # split into input and output elements
    # data = dataframe.values
    # data = data.astype('float32')
    # X, y = data[:, :-1], data[:, -1]
    # print(X.shape, y.shape)

    # separate into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=1)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # define the search
    search = StructuredDataRegressor(max_trials=15, loss='mean_absolute_error')

    # perform the search
    search.fit(x=X_train, y=y_train, verbose=1)

    # evaluate the model
    mae, _ = search.evaluate(X_test, y_test, verbose=0)
    print('MAE: %.3f' % mae)

    # # use the model to make a prediction
    # X_new = asarray([[108]]).astype('float32')
    # yhat = search.predict(X_new)
    # print('Predicted: %.3f' % yhat[0])]

    # get the best performing model
    model = search.export_model()
    # summarize the loaded model
    model.summary()
    # save the best performing model to file
    #model.save('model_insurance.h5', save_format="h5")

    predictions = model.predict(test)
    preds = []
    for pred in predictions:
        preds.append(pred[0])

    res = pd.DataFrame()
    res['num_sold'] = preds
    res.to_csv("predict_fnn_v1.csv")


def test_1():
    model = load_model('model_insurance.h5')
    predictions = model.predict(test)
    preds = []
    for pred in predictions:
        preds.append(pred[0])

    res = pd.DataFrame()
    res['target'] = preds
    res.to_csv("auto_keras_v2.csv")

train_1()