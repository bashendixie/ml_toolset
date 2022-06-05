import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

train_dir = 'paddy-disease-classification/train_images/'
batch_size = 8
img_dim = 128

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

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(img_dim, img_dim, 3)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu' ),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(N_CLASS, activation='softmax')
])


model.summary()
tf.keras.utils.plot_model(model, to_file='model.png',expand_nested=True,show_shapes=True)

EPOCH = 100

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=5,
                            verbose=0,
                            mode='auto')

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                            factor=0.5,
                            patience=5)

callbacks = [early_stopping_callback, lr_reducer]


with tf.device(tf.test.gpu_device_name()):
    model.compile(
        #optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
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

model.save("study_self_1.h5")

# load_model("study_self_1.h5")

test_path = 'paddy-disease-classification/test_images'
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

results=pd.DataFrame({"image_id":filenames,
                      "label":predictions})
results.image_id = results.image_id.str.replace('./', '')
results.to_csv("submission.csv",index=False)
results.head()

results['label'].value_counts()






