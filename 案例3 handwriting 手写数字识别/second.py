import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from keras.models import load_model
import cv2
import tensorflow_hub as hub
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

#保存部分训练图片
def saveImg():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('D:\\Project\\DeepLearn\\1dataset\\1\\mnist.npz')

    # 保存前20张图片
    for i in range(50):
        show_pic = x_train[i].copy()
        cv2.imwrite('C:\\Users\\zyh\\Desktop\\1\\' + str(i) +'.jpg',show_pic)

#训练模型并保存h5
def train():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('D:\\Project\\DeepLearn\\1dataset\\1\\mnist.npz')
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28)),
            # Use a Rescaling layer to make sure input values are in the [0, 1] range.
            layers.experimental.preprocessing.Rescaling(1.0 / 255),
            # The original images have shape (28, 28), so we reshape them to (28, 28, 1)
            layers.Reshape(target_shape=(28, 28, 1)),
            # Follow-up with a classic small convnet
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(32, 3, activation="relu"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=keras.metrics.SparseCategoricalAccuracy(),
    )
    model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1)
    model.save('C:\\Users\\zyh\\Desktop\\1\\mnist.h5')

#h5转pb
def h5_to_pb():
    model = tf.keras.models.load_model('C:\\Users\\zyh\\Desktop\\1\\mnist.h5',
                                       custom_objects={'KerasLayer': hub.KerasLayer, 'Dense': tf.keras.layers.Dense},
                                       compile=False)
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
#测试模型
def test():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(
        'D:\\Project\\DeepLearn\\1dataset\\1\\mnist.npz')

    net = load_model('C:/Users/zyh/Desktop/1/mnist.h5')
    pred_image = cv2.imread("C:/Users/zyh/Desktop/1/12345.jpg", cv2.IMREAD_GRAYSCALE)
    #pred_image = 255 - pred_image
    pred_image = tf.reshape(pred_image, (1, 28*28))
    #pred_image = tf.reshape(pred_image, (-1, 28 * 28))
    pred = net.predict(pred_image)
    print(f'pred: {tf.argmax(pred, axis=1).numpy()}')

#train()
test()
#saveImg()
#h5_to_pb()