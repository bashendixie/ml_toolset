from lenet import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# grab the MNIST dataset (if this is your first time using this
# dataset then the 11MB download may take a minute)
print("[INFO] accessing MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# if we are using "channels first" ordering, then reshape the
# design matrix such that the matrix is:
# num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
	trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
	testData = testData.reshape((testData.shape[0], 1, 28, 28))
# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
	trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
	testData = testData.reshape((testData.shape[0], 28, 28, 1))

# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0
# convert the labels from integers to vectors
le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

# 合并数据，用所有数据进行训练
merge_data = np.append(trainData, testData, axis=0)
merge_label = np.append(trainLabels, testLabels, axis=0)


# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.001)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# train the network
print("[INFO] training network...")

# 用所有数据进行训练, 没有验证集
H = model.fit(merge_data, merge_label, batch_size=128, epochs=300, verbose=1)
# 用60000数据进行训练， 10000进行验证
# H = model.fit(trainData, trainLabels, validation_data=(testData, testLabels), batch_size=128, epochs=50, verbose=1)
# 保存模型
model.save("lenet_mnist.v9.h5")

# 验证
print("[INFO] evaluating network...")
predictions = model.predict(testData, batch_size=128)
print(classification_report(testLabels.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in le.classes_]))

# 绘制损失和准确率
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 300), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 300), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 300), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 300), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
