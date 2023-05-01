from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
import cv2

data = pd.read_csv('origin/test.csv')
#data.drop_duplicates(keep='first', inplace=True)
print(data.shape)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

index = 0
for row in data.values:
    randomByteArray = []
    for i in range(286):
        randomByteArray.append(row[1:287])
    flatNumpyArray = np.array(randomByteArray)
    flatNumpyArray = normalization(flatNumpyArray) * 255
    grayImage = flatNumpyArray.reshape(286, 286)
    # show gray images
    path = 'to_img/test/' + str(int(row[0])) + '.jpg'
    print(path)
    cv2.imwrite(path, grayImage)
    #cv2.imshow('GrayImage', grayImage)
    index = index+1



