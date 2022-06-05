from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import argparse
import pandas as pd
import os
import cv2
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_dir = '../input/paddyorigin/train_images'

batch_size = 128
img_dim = 224

img_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.1,
    rotation_range=5,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
)

train_gen = img_datagen.flow_from_directory(
    train_dir,
    subset="training",
    seed=42,
    target_size=(img_dim, img_dim),
    batch_size=batch_size,
    class_mode="categorical",
)

valid_gen = img_datagen.flow_from_directory(
    train_dir,
    subset="validation",
    seed=42,
    target_size=(img_dim, img_dim),
    batch_size=batch_size,
    class_mode="categorical",
)

print(train_gen.class_indices)
print(len(train_gen.class_indices))
N_CLASS = len(train_gen.class_indices)

model = VGG16(weights=None, classes=10, input_tensor=Input(shape=(img_dim, img_dim, 3)))
model.summary()
tf.keras.utils.plot_model(model, to_file='model.png',expand_nested=True,show_shapes=True)

EPOCH = 100
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=0,
                            mode='auto')

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                            factor=0.5,
                            patience=5)

checkpoint_path = './'
os.makedirs(checkpoint_path, exist_ok=True)
model_path = os.path.join(checkpoint_path, 'study_VGG16_1.h5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path,onitor='val_loss',save_best_only=True, verbose=1)

callbacks = [early_stopping_callback, lr_reducer, checkpoint]


with tf.device(tf.test.gpu_device_name()):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history_model = model.fit(
        train_gen,
        epochs=EPOCH,
        validation_data = valid_gen,
        callbacks=callbacks,
    )

pd.DataFrame(history_model.history).plot(figsize=(8,5))
plt.show()


test_path = '../input/paddyorigin/test_images'
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255).flow_from_directory(
    directory=test_path,
    target_size=(img_dim, img_dim),
    batch_size=batch_size,
    classes=['.'],
    shuffle=False,
)

predict = model.predict(test_gen, verbose=1)

predicted_class_indices=np.argmax(predict,axis=1)
print(set(predicted_class_indices))

inv_map = {v:k for k,v in train_gen.class_indices.items()}

predictions = [inv_map[k] for k in predicted_class_indices]

filenames=test_gen.filenames

results=pd.DataFrame({"image_id":filenames, "label":predictions})
results.image_id = results.image_id.str.replace('./', '')
results.to_csv("./submission_3.csv",index=False)











