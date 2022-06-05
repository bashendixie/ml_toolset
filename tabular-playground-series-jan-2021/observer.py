# importing packages
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import cv2

# 分割成多个csv
def read_and_split():
    index = 0
    df = pd.read_csv('group/train.csv')
    upper = np.where(df['target1'] == index)
    df.loc[upper].to_csv('group/' +str(index)+ '.csv')

# 循环删除异常数据
def del_execption():
    arr = [0,4,5,6,7,8,9,10]
    ind = ['cont1','cont2','cont3','cont4','cont5','cont6','cont7',
           'cont8','cont9','cont10','cont11','cont12','cont13','cont14']
    for i in arr:
        for title in ind:
            path = 'group/' + str(i) + '.csv'
            df = pd.read_csv(path)

            sns.boxplot(y=df[title], data=df)
            plt.show()

            # IQR
            Q1 = np.percentile(df[title], 25, interpolation='midpoint')
            Q3 = np.percentile(df[title], 75, interpolation='midpoint')
            IQR = Q3 - Q1

            print("Old Shape: ", df.shape)
            # Upper bound
            upper = np.where(df[title] >= (Q3 + 1.5 * IQR))
            # Lower bound
            lower = np.where(df[title] <= (Q1 - 1.5 * IQR))

            # Removing the Outliers
            if len(upper[0])>1:
                df.drop(upper[0], inplace=True)

            if len(lower[0]) > 1:
                df.drop(lower[0], inplace=True)

            print("New Shape: ", df.shape)
            # sns.boxplot(x=df[title], data=df)
            sns.boxplot(y=df[title], data=df)
            plt.show()
            if len(upper[0]) > 1 or len(lower[0]) > 1:
                df.to_csv(path)



def del_merge_data():
    arr = [4, 5, 6, 7, 8, 9, 10]
    df0 = pd.read_csv('group/0.csv')
    df4 = pd.read_csv('group/4.csv')
    df5 = pd.read_csv('group/5.csv')
    df6 = pd.read_csv('group/6.csv')
    df7 = pd.read_csv('group/7.csv')
    df8 = pd.read_csv('group/8.csv')
    df9 = pd.read_csv('group/9.csv')
    df10 = pd.read_csv('group/10.csv')
    df0 = df0.append(df4)
    df0 = df0.append(df5)
    df0 = df0.append(df6)
    df0 = df0.append(df7)
    df0 = df0.append(df8)
    df0 = df0.append(df9)
    df0 = df0.append(df10)
    df0.to_csv('group/merge.csv')

#del_merge_data()


# 删除异常数据
def del_execption1():
    path = 'group/9.csv'
    # Load the dataset
    df = pd.read_csv(path) #, index_col='id'

    title = "cont1"
    target = "target"

    # df = sns.load_dataset('origin/train.csv')
    # df.head()
    # #
    #sns.boxplot(x=df[title], data=df)
    sns.boxplot(y=df[title], x=round(df[target]), data=df)
    plt.show()


    # IQR
    Q1 = np.percentile(df[title], 25, interpolation='midpoint')
    Q3 = np.percentile(df[title], 75, interpolation='midpoint')
    IQR = Q3 - Q1

    print("Old Shape: ", df.shape)
    # Upper bound
    upper = np.where(df[title] >= (Q3 + 1.5 * IQR))
    # Lower bound
    lower = np.where(df[title] <= (Q1 - 1.5 * IQR))
    # Removing the Outliers
    df.drop(upper[0], inplace=True)
    df.drop(lower[0], inplace=True)

    print("New Shape: ", df.shape)
    #sns.boxplot(x=df[title], data=df)
    sns.boxplot(y=df[title], x=round(df[target]), data=df)
    plt.show()

    df.to_csv(path)


    #
    # def graph(y):
    #     sns.boxplot(y=y, data=train)
    #
    # plt.figure(figsize=(10, 10))
    #
    # # Adding the subplot at the specified
    # # grid position
    # plt.subplot(221)
    # graph('cont2')
    #
    # plt.show()


def del_read_all():
    df = pd.read_csv('C:\\Users\\zyh\\Desktop\\customers.csv')
    del df['FN']
    del df['Active']
    del df['club_member_status']
    del df['fashion_news_frequency']
    del df['age']
    del df['postal_code']

    df['prediction'] = ['0118458029 0145872001 0189616008 0189626001 0194242047 0194242048 0212629032 0237347011 0241602018 0252298005 0252298013 0620255001' for i in df["customer_id"]]
    #for index, row in df.iterrows():
    #    row['prediction'] = '0118458029 0145872001 0189616008 0189626001 0194242047 0194242048 0212629032 0237347011 0241602018 0252298005 0252298013 0620255001'

    df.to_csv('C:\\Users\\zyh\\Desktop\\2022030901.csv', index=0)

# del_read_all()

def h5_to_pb():

    #tf.compat.v1
    model = tf.compat.v1.keras.models.load_model('C:\\Users\\zyh\\Desktop\\unet_meiyan_v1.h5',
                                       custom_objects={'KerasLayer': hub.KerasLayer, 'Dense': tf.compat.v1.keras.layers.Dense},
                                       compile=False)

    #cv2.dnn.writeTextGraph('C:\\Users\\zyh\\Desktop\\0001.pb', 'C:\\Users\\zyh\\Desktop\\graph.pbtxt')

    model.summary()
    full_model = tf.compat.v1.function(lambda Input: model(Input))

    full_model = full_model.get_concrete_function(tf.compat.v1.TensorSpec(model.inputs[0].shape, tf.compat.v1.float32))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.compat.v1.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="D:\\", name="0004.pb", as_text=False)


def h5_to_pb1():
    model = tf.compat.v1.keras.models.load_model('C:\\Users\\zyh\\Desktop\\unet_meiyan_v1.h5', compile=False)

    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="D:\\",
                      name="model123.pb",
                      as_text=False)

h5_to_pb()