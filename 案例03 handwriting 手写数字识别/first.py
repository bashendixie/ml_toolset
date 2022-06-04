# import cv2
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
# import matplotlib.pyplot as plt
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# print('tf版本：',tf.__version__)
#
# # 归一化
# def prepprocess(x,y):
#     # 归一化
#     x = tf.cast(x, dtype=tf.float32) / 255.
#     x = tf.reshape(x, [28 * 28])
#     y = tf.cast(y, dtype=tf.int32)
#     y = tf.one_hot(y, depth=10)
#     return x, y
#
# #训练
# def train():
#     # 构建网络
#     network = keras.Sequential([
#         keras.layers.Dense(128,'relu'),
#         keras.layers.Dense(10)
#     ])
#     network.build(input_shape=(None,28*28))
#     network.summary()
#
#     network.compile(optimizer=keras.optimizers.Adam(lr=0.01),
#                   loss = tf.losses.CategoricalCrossentropy(from_logits=True),
#                   # loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),      # 用这个不用tf.one_hot()
#                   metrics=['accuracy']
#     )
#     # 训练
#     history = network.fit(train_data,epochs=15,validation_data=test_data,validation_freq=1)
#     plt.plot(history.history['accuracy'],label='accuracy')
#     plt.plot(history.history['val_accuracy'],label='val_accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('accuracy')
#     plt.ylim([0.5,1])
#     plt.legend(loc='lower right')
#     plt.show()
#     tf.saved_model.save(network,'C:\\Users\\zyh\\Desktop\\1\\')
#     print("保存模型成功")
#
#     # Convert Keras model to ConcreteFunction
#     full_model = tf.function(lambda x: network(x))
#     #full_model = full_model.get_concrete_function(tf.TensorSpec(network.inputs[0].shape, network.inputs[0].dtype))
#     full_model.get_concrete_function(tf.TensorSpec(network.inputs[0].shape, tf.float32))
#
#     # Get frozen ConcreteFunction
#     frozen_func = convert_variables_to_constants_v2(full_model)
#     frozen_func.graph.as_graph_def()
#
#     layers = [op.name for op in frozen_func.graph.get_operations()]
#     print("-" * 50)
#     print("Frozen model layers: ")
#     for layer in layers:
#         print(layer)
#
#     print("-" * 50)
#     print("Frozen model inputs: ")
#     print(frozen_func.inputs)
#     print("Frozen model outputs: ")
#     print(frozen_func.outputs)
#
#     # Save frozen graph from frozen ConcreteFunction to hard drive
#     tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir="C:\\Users\\zyh\\Desktop\\1\\", name="frozen_graph.pb", as_text=False)
#
#
# # 获取数据集
# (train_image, train_label), (test_image, test_label) = keras.datasets.mnist.load_data('D:\\Project\\DeepLearn\\1dataset\\mnist.npz')
# print("数据维度(训练集)：", train_image.shape, train_label.shape)
# print("数据维度(测试集)：", test_image.shape, test_label.shape)
#
# train_data = tf.data.Dataset.from_tensor_slices((train_image, train_label))
# train_data = train_data.map(prepprocess).batch(128)
# test_data = tf.data.Dataset.from_tensor_slices((test_image, test_label))
# test_data = test_data.map(prepprocess).batch(128)
# class_name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
# # 数据可视化
# plt.figure(figsize=(10, 10))
# for i in range(25):
# 	plt.subplot(5, 5, i + 1)
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.grid(False)
# 	plt.imshow(train_image[i], cmap=plt.cm.binary)
# 	plt.xlabel(class_name[train_label[i]])
# plt.show()
#
# train()




import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Sequential, optimizers, metrics
from keras.models import load_model
import numpy as np
import tensorflow_hub as hub
import cv2
import os
from tensorflow.python.keras import backend as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


# 屏蔽通知信息和警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 每批几张图片
batch_size = 2
# 类别数
class_num = 10
# 保存模型的路径
save_path = "D:\\Project\\DeepLearn\\1dataset\\1\\mnist.h5"
# pb路径
output_graph = "C:\\Users\\zyh\\Desktop\\1\\mnist"
# 导出pb
writepb = False
# 展示样例
show_demo = False
# 验证测试集
evaluate_dataset = True
# 是否训练
run_train = False
# 图片路径, 仅用于 detect_image(), 当为False时不识别
image_path = 'images/36.png'


# 读取本地数据集
(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data('D:\\Project\\DeepLearn\\1dataset\\1\\mnist.npz')

if show_demo:
    print(f'dataset info: shape: {x_train.shape}, {y_train.shape}')
    print(f'dataset info: max: {x_train.max()}')
    print(f'dataset info: min: {x_train.min()}')

    print("A sample:")
    print("y_train: ", y_train[0])
    # print("x_train: \n", x_train[0])
    show_pic = x_train[0].copy()
    show_pic = cv2.resize(show_pic, (28 * 10, 28 * 10))
    cv2.imshow("A image sample", show_pic)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()
        print("show demo over")

# convert to tf tensor 转换为tf 数据集的格式, 并进行归一化
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) // 255.
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) // 255.
dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset_train = dataset_train.batch(batch_size).repeat(class_num)

# 在这里定义一个简单的全连接网络
def build_simple_net():
    net = Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(class_num)
    ])
    net.build(input_shape=(None, 28 * 28))
    # net.summary()
    return net

# 使用 SGD 优化器进行训练
def train(print_info_step=250):
    net = build_simple_net()
    # 优化器
    optimizer = optimizers.SGD(lr=0.01)
    # 计算准确率
    acc = metrics.Accuracy()

    for step, (x, y) in enumerate(dataset_train):
        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28 * 28))
            # [b, 784] => [b, 10]
            out = net(x)
            # [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=class_num)
            # [b, 10]
            loss = tf.square(out - y_onehot)
            # [b]
            loss = tf.reduce_sum(loss) / batch_size

        # 反向传播
        acc.update_state(tf.argmax(out, axis=1), y)
        grads = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))

        if acc.result() >= 0.90:
            net.save(save_path)
            h5_to_pb(save_path)
            #model = load_model(save_path)

            #model = tf.keras.Model.load_weights(save_path)
            #frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
            #tf.train.write_graph(frozen_graph, "model", output_graph+"\\tf_model.pb", as_text=False)
            #tf.train.write_graph(frozen_graph, "model", output_graph + "\\tf_model.pbtxt", as_text=False)
            #net.save(save_path)
            # tf.keras.models.save_model(net, output_graph, True, False, 'tf')
            #
            # v = tf.Variable(0, name='my_variable')
            # sess = tf.compat.v1.Session()
            # tf.io.write_graph(sess.graph_def, output_graph, 'train.pbtxt')
            print(f'final acc: {acc.result()}, total step: {step}')
            break

        if step % print_info_step == 0:
            print(f'step: {step}, loss: {loss}, acc: {acc.result().numpy()}')
            acc.reset_states()

        if step % 500 == 0 and step != 0:
            print('save model')
            #net.save(save_path)

def h5_to_pb(h5_save_path):

    model = tf.keras.models.load_model(h5_save_path, custom_objects={'KerasLayer': hub.KerasLayer, 'Dense':tf.keras.layers.Dense}, compile=False)
    # model = tf.keras.models.load_model(h5_save_path, compile=False)
    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    # 下面这句话出了问题
    # full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # 改成了这样子的,如愿生成了pb文件
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, tf.float32))

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
                      logdir="C:\\Users\\zyh\\Desktop\\1\\",
                      name="model.pb",
                      as_text=False)

def test_dataset():
    net = build_simple_net()
    # 加载模型
    net.load_weights(save_path)
    # 拿到测试集第一张图片
    pred_image = cv2.imread("D:/Project/DeepLearn/1dataset/1/4.png")#x_test[0]
    pred_image = tf.reshape(pred_image, (-1, 28 * 28))
    pred = net.predict(pred_image)
    print(f'pred: {tf.argmax(pred, axis=1).numpy()}')

def split_number(img):
    result = []
    net = build_simple_net()
    # 加载模型
    net.load_weights(save_path)

    image = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    for cnt in contours[:-1]:
        x, y, w, h = cv2.boundingRect(cnt)

        image = img[y:y+h, x:x+w]
        image = cv2.resize(image, (28, 28))

        pred_image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.
        pred_image = tf.reshape(pred_image, (-1, 28 * 28))
        pred = net.predict(pred_image)
        out = tf.argmax(pred, axis=1).numpy()
        result = [out[0]] + result
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("demo", img)
    print(result)
    k = cv2.waitKey(0)
    if k == ord('q'):
        pass
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if run_train:
        train()
    elif evaluate_dataset:
        test_dataset()
    elif image_path:
        image = cv2.imread(image_path)
        # detect_image(image)
        split_number(image)