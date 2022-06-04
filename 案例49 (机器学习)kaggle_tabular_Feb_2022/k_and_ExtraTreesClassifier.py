from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import  StratifiedKFold,KFold
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
from statistics import mean
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate

np.random.seed(228)
tf.random.set_seed(228)

pd.set_option('display.max_columns', None)

#########################################################

train = pd.read_csv('origin/train.csv', index_col='row_id')
test = pd.read_csv('origin/test.csv', index_col='row_id')
ss = pd.read_csv('origin/sample_submission.csv')


#########################################################

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
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


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)




features = train.columns.tolist()[0:-1]
# sample_weight = train['sample_weight']

train['std'] = train[features].std(axis = 1)
test['std'] = test[features].std(axis = 1)

train['min'] = train[features].min(axis = 1)
test['min'] = test[features].min(axis = 1)

train['max'] = train[features].max(axis = 1)
test['max'] = test[features].max(axis = 1)

features += ['std', 'min', 'max']

le = LabelEncoder()
train['target'] = le.fit_transform(train['target'])

sc = StandardScaler()
train[features] = sc.fit_transform(train[features])
test[features] = sc.transform(test[features])

X = train[features]
y = train['target']



predictions, scores = [], []

k = StratifiedKFold(n_splits=100, random_state=228, shuffle=True)
for i, (trn_idx, val_idx) in enumerate(k.split(X, y)):
    X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    model = ExtraTreesClassifier(n_estimators=1111, n_jobs=4)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_score = accuracy_score(y_val, val_pred)
    print(f'Fold {i + 1} accuracy score: {round(val_score, 4)}')

    scores.append(val_score)
    predictions.append(model.predict_proba(test))
print('')
print(f'Mean accuracy - {round(mean(scores), 4)}')

y_proba = sum(predictions) / len(predictions)
#y_proba += np.array([0, 0, 0.025, 0.045, 0, 0, 0, 0, 0, 0])
y_pred_tuned = le.inverse_transform(np.argmax(y_proba, axis=1))
pd.Series(y_pred_tuned, index=test.index).value_counts().sort_index() / len(test) * 100

ss['target'] = y_pred_tuned
ss.to_csv('submission_v100.csv', index=False)