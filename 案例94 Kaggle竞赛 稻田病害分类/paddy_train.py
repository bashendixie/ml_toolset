import tensorflow as tf
from paddy_fcn_model import FCN_model
from paddy_generator import Generator
import os


def train(model, train_generator, val_generator, epochs=50):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_path = './snapshots'
    os.makedirs(checkpoint_path, exist_ok=True)
    model_path = os.path.join(checkpoint_path,
                              'model_epoch_2.h5')

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=len(train_generator),
                                  epochs=epochs,
                                  callbacks=[tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss',
                                                                                save_best_only=True, verbose=1)],
                                  validation_data=val_generator,
                                  validation_steps=len(val_generator))

    return history


if __name__ == "__main__":
    # Create FCN model
    model = FCN_model(len_classes=10, dropout_rate=0.2)
    model.load_weights('model_epoch.h5')

    # The below folders are created using utils.py
    train_dir = 'data_set/train'
    val_dir = 'data_set/val'

    # If you get out of memory error try reducing the batch size
    BATCH_SIZE = 4
    train_generator = Generator(train_dir, BATCH_SIZE, shuffle_images=True, image_min_side=24)
    val_generator = Generator(val_dir, BATCH_SIZE, shuffle_images=True, image_min_side=24)

    EPOCHS = 50
    history = train(model, train_generator, val_generator, epochs=EPOCHS)