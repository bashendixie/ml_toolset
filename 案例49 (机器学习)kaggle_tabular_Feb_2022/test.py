import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import os
import cv2
import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
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
#return week_day_dict[day]


def test0():
    print("这是test0")
    model = load_model("tabular_fnn_dup_addlie_1.h5")
    model.summary()

    data = pd.read_csv('data/test.csv')
    #cols = ['date', 'Weekday', 'Festival', 'country_Finland', 'country_Norway', 'country_Sweden', 'store_KaggleMart', 'store_KaggleRama', 'product_Kaggle Hat', 'product_Kaggle Mug', 'product_Kaggle Sticker']

    #data = data * 1000000
    # 归一化
    data = (data - data.min()) / (data.max() - data.min())
    # 增加和、平均、最大、最小的列
    res = pd.DataFrame()
    res['sum'] = data.sum(axis=1)
    res['max'] = data.max(axis=1)
    res['min'] = data.min(axis=1)
    res['mean'] = data.mean(axis=1)

    data['sum'] = res['sum']
    data['max'] = res['max']
    data['min'] = res['min']
    data['mean'] = res['mean']


    predictions = model.predict(data, batch_size=10000)

    preds = []
    for pred in predictions:
        preds.append(week_day_dict[np.argmax(pred)])

    print(preds)

    res = pd.DataFrame()
    res['target'] = preds
    res.to_csv("tabular_fnn_dup_addlie_1.csv")


def test1():
    print("这是test1")
    model = load_model("tabular_fnn_2.h5")
    model.summary()

    test = pd.read_csv('data/test_fnn.csv')
    #cols = ['date', 'Weekday', 'Festival', 'country_Finland', 'country_Norway', 'country_Sweden', 'store_KaggleMart', 'store_KaggleRama', 'product_Kaggle Hat', 'product_Kaggle Mug', 'product_Kaggle Sticker']
    cols = ['date', 'week', 'festival', 'country', 'store', 'product']
    tests = test[cols]#.head()
    #tests = test.to_numpy()
    #tests = tests.astype('float64')

    predictions = model.predict(tests, batch_size=6570)
    print(predictions)

    test['num'] = predictions
    test.to_csv("predict_v15.csv")


def test2():
    print("这是test2")
    model = load_model("tabular_13_3.h5")
    model.summary()

    test = pd.read_csv('data/test_k.csv')
    # cols = ['date', 'Weekday', 'Festival', 'country_Finland', 'country_Norway', 'country_Sweden', 'store_KaggleMart', 'store_KaggleRama', 'product_Kaggle Hat', 'product_Kaggle Mug', 'product_Kaggle Sticker']
    #cols = ['date', 'week', 'festival', 'country', 'store', 'product']
    #tests = test[cols]  # .head()

    tests = test.to_numpy()
    tests = tests.astype('float64')

    predictions = model.predict(tests, batch_size=6570)
    print(predictions)

    test['num'] = predictions
    test.to_csv("predict_v16.csv")


def test_for_xgboost():
    print("这是xgboost")
    f2 = open('xgboost.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)
    test_X = pd.read_csv('data/test.csv')
    predictions = model1.predict(test_X)
    preds = []
    for pred in predictions:
        preds.append(week_day_dict[pred])

    res = pd.DataFrame()
    res['target'] = preds
    res.to_csv("predict_xgboost_v1.csv")


def test_for_gcforest():
    print("这是gcforest")
    f2 = open('gcforest.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)
    test_X = pd.read_csv('data/test.csv')

    predictions = model1.predict(test_X.to_numpy())
    preds = []
    for pred in predictions:
        preds.append(week_day_dict[pred])

    res = pd.DataFrame()
    res['target'] = preds
    res.to_csv("predict_gcforest_v1.csv")



def test_for_deepforest():
    print("这是deepforest")
    f2 = open('deepforest_v3.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)
    test_X = pd.read_csv('data/test.csv')

    predictions = model1.predict(test_X.to_numpy())
    preds = []
    for pred in predictions:
        preds.append(week_day_dict[pred])

    res = pd.DataFrame()
    res['target'] = preds
    res.to_csv("predict_deepforest_v3.csv")


def test_for_randomforest():
    print("这是randomforest")
    f2 = open('random_forest.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)
    test_X = pd.read_csv('data/test.csv')

    predictions = model1.predict(test_X)
    preds = []
    for pred in predictions:
        preds.append(week_day_dict[pred])

    res = pd.DataFrame()
    res['target'] = preds
    res.to_csv("predict_randomforest_v1.csv")


def test_for_extratrees():
    print("这是extratrees")
    f2 = open('extra_trees_train_and_test_dup_v1.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)
    test_X = pd.read_csv('data/test.csv')

    predictions = model1.predict(test_X)
    preds = []
    for pred in predictions:
        preds.append(week_day_dict[pred])

    res = pd.DataFrame()
    res['target'] = preds
    res.to_csv("extra_trees_train_and_test_dup_v1.csv")

def test_for_StackingClassifier():
    print("这是StackingClassifier")
    f2 = open('StackingClassifier_v2.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)
    test_X = pd.read_csv('data/test.csv')

    predictions = model1.predict(test_X)
    preds = []
    for pred in predictions:
        preds.append(week_day_dict[pred])

    res = pd.DataFrame()
    res['target'] = preds
    res.to_csv("predict_StackingClassifier_v3.csv")

def test_for_BaggingClassifier():
    print("这是BaggingClassifier")
    f2 = open('BaggingClassifier_v4.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)
    test_X = pd.read_csv('data/test.csv')

    predictions = model1.predict(test_X)
    preds = []
    for pred in predictions:
        preds.append(week_day_dict[pred])

    res = pd.DataFrame()
    res['target'] = preds
    res.to_csv("predict_BaggingClassifier_v4.csv")


def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist

def testmodel_with_():
    model = load_model('tabular_vgg_4.h5')
    imagePaths = []
    preds = []
    paths = 'to_img/test/'
    # grab the images paths and randomly shuffle them
    imagePaths = sorted(list(getFileList(paths, imagePaths)))
    for imagePath in imagePaths:
        # load the input images and resize it to the target spatial dimensions
        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        image = image.astype("float") / 255.0
        image = image.reshape((1, 286, 286))
        print(image.shape)
        #images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
        pred = model.predict(image)
        i = pred.argmax(axis=1)[0]
        preds.append(week_day_dict[i])
        print(imagePath)
        print(str(i))

    res = pd.DataFrame()
    res['target'] = preds
    res.to_csv("predict_SmallVGGNet_v2.csv")


def test_for_lightgbm():
    print("这是lightgbm")
    f2 = open('lightgbm_v2.model', 'rb')
    s2 = f2.read()
    model1 = pickle.loads(s2)
    test_X = pd.read_csv('data/test.csv')

    predictions = model1.predict(test_X)
    preds = []
    for pred in predictions:
        preds.append(week_day_dict[pred])

    res = pd.DataFrame()
    res['target'] = preds
    res.to_csv("predict_lightgbm_v2.csv")

test_for_lightgbm()