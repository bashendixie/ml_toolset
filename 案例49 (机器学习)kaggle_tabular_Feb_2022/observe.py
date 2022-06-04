import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D,Conv1D,Embedding
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K


week_day_dict = {
    0 : 'Streptococcus_pyogenes',
    1 : 'Salmonella_enterica',
    2 : 'Enterococcus_hirae',
    3 : 'Escherichia_coli',
    4 : 'Campylobacter_jejuni',
    5 : 'Streptococcus_pneumoniae',
    6 : 'Staphylococcus_aureus',
    7 : 'Escherichia_fergusonii',
    8 : 'Bacteroides_fragilis',
    9 : 'Klebsiella_pneumoniae',
}


data = pd.read_csv('data/train.csv')
del data['row_id']
# 去除重复
duplicates_train = data.duplicated().sum()
print('Duplicates in train data: {0}'.format(duplicates_train))

data.drop_duplicates(keep='first', inplace=True)
duplicates_train = data.duplicated().sum()

print('Train data shape:', data.shape)
print('Duplicates in train data: {0}'.format(duplicates_train))

#s_bool = data['target'] ==week_day_dict[9]
#print(s_bool.sum())


