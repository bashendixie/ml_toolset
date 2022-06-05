import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import sys
import math


sequence_length = 32
batch_size = 1024
categorical_columns = ["x", "y", "direction", "hour", "month", "dayofweek"]
sequence_categorical_columns = ["x", "y", "direction"]
is_training = True


train = pd.read_csv("input/tabular-playground-series-mar-2022/train.csv")
direction_map = dict()
for i, direction in enumerate(train.direction.unique()):
    direction_map[direction] = i


def feature_engineering(data):
    data["key"] = data["x"].map(lambda item: str(item)) + "_" + data["y"].map(lambda item: str(item)) + "_" + data["direction"]
    data["direction"] = data["direction"].map(lambda item: direction_map[item])
    data['time'] = pd.to_datetime(data['time'])
    data['month'] = data['time'].dt.month
    data['dayofweek'] = data['time'].dt.dayofweek
    data['hour'] = data['time'].dt.hour
    data = data.drop(['time'], axis=1)
    data["x"] = data["x"].astype(np.float32)
    data["y"] = data["y"].astype(np.float32)
    return data


train = feature_engineering(train)
train.head(30)

set(train["key"].value_counts())

train[train.key=="0_0_EB"].congestion.plot()

train[train.key=="0_0_EB"].congestion[-100:-1].plot()

def preprocess(window):
    return (
        window[:-1, 0],
        window[:-1, 1],
        window[:-1, 2],
        window[:-1, 3],
        window[-1, 0],
        window[-1, 1],
        window[-1, 2],
        window[-1, 3],
        window[-1, 4],
        window[-1, 5],
    ), window[-1:, -1]

def make_dataset(df, sequence_length=32, mode="train"):
    dataset = tf.data.Dataset.from_tensor_slices((df[categorical_columns + ["congestion"]]))
    dataset = dataset.window(sequence_length + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(sequence_length + 1))
    dataset = dataset.map(preprocess)
    if mode == "train":
        dataset = dataset.shuffle(buffer_size=batch_size)
    dataset = dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return dataset


split_fraction = 0.9
split_index = int(len(train) * split_fraction)
train_data = train[0:split_index]
val_data = train[split_index:]
train_data.shape, val_data.shape
train_ds = make_dataset(train_data)
valid_ds = make_dataset(val_data, mode="valid")


lookupLayersMap = dict()
for column in categorical_columns:
    unique_values = list(train[column].unique())
    lookupLayersMap[column] = tf.keras.layers.IntegerLookup(vocabulary=unique_values)


def get_model():
    sequence_inputs = []
    sequence_vectors = []
    dense_inputs = []
    dense_vectors = []
    for column in sequence_categorical_columns:
        sequence_input = tf.keras.Input(shape=(sequence_length, 1), name=f"{column}_sequnce_input", dtype=tf.int64)
        lookup = lookupLayersMap[column]
        vocab_size = len(lookup.get_vocabulary())
        embed_dimension = max(math.ceil(np.sqrt(vocab_size)), 2)
        print(column)
        sequence_vector = lookup(sequence_input)
        sequence_vector = tf.keras.layers.Embedding(vocab_size, embed_dimension, input_length=sequence_length)(
            sequence_vector)
        sequence_vector = tf.keras.layers.Reshape((-1, embed_dimension))(sequence_vector)
        sequence_vectors.append(sequence_vector)
        sequence_inputs.append(sequence_input)
    target_sequence_input = tf.keras.Input(shape=(sequence_length, 1))
    sequence_inputs.append(target_sequence_input)
    sequence_vectors.append(target_sequence_input)
    sequence_vector = tf.keras.layers.Concatenate(axis=-1)(sequence_vectors)
    sequence_vector = tf.keras.layers.LSTM(128, return_sequences=True)(sequence_vector)
    sequence_vector = tf.keras.layers.LSTM(64, return_sequences=False)(sequence_vector)
    sequence_vector = tf.keras.layers.Dense(128, activation="relu")(sequence_vector)
    sequence_vector = tf.keras.layers.Dense(128, activation="relu")(sequence_vector)
    sequence_vector = tf.keras.layers.Dense(128, activation="relu")(sequence_vector)
    sequence_vector = tf.keras.layers.Dense(128, activation="relu")(sequence_vector)

    for column in categorical_columns:
        dense_input = tf.keras.Input(shape=(1,), name=f"{column}_dense_input")
        lookup = lookupLayersMap[column]
        vocab_size = len(lookup.get_vocabulary())
        embed_dimension = max(math.ceil(np.sqrt(vocab_size)), 2)
        dense_vector = lookup(dense_input)
        dense_vector = tf.keras.layers.Embedding(vocab_size, embed_dimension, input_length=1)(dense_vector)
        dense_vector = tf.keras.layers.Reshape((-1,))(dense_vector)
        dense_vectors.append(dense_vector)
        dense_inputs.append(dense_input)

    dense_vector = tf.keras.layers.Concatenate(axis=-1)(dense_vectors)
    dense_vector = tf.keras.layers.Dense(128, activation="relu")(dense_vector)
    dense_vector = tf.keras.layers.Dense(128, activation="relu")(dense_vector)
    dense_vector = tf.keras.layers.Dense(128, activation="relu")(dense_vector)
    dense_vector = tf.keras.layers.Dense(128, activation="relu")(dense_vector)

    vector = tf.keras.layers.Concatenate(axis=-1)([sequence_vector, dense_vector])
    vector = tf.keras.layers.Dense(32, activation="relu")(vector)
    output = tf.keras.layers.Dense(1)(vector)
    model = tf.keras.Model(inputs=sequence_inputs + dense_inputs, outputs=output)
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return model


model = get_model()
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

cp = tf.keras.callbacks.ModelCheckpoint("model.tf", monitor="val_mae", save_best_only=True, save_weights_only=True)
es = tf.keras.callbacks.EarlyStopping(patience=10)
if is_training:
    model.fit(train_ds, epochs=50, validation_data=valid_ds, callbacks=[es, cp])
    model.load_weights("model.tf")
else:
    model.load_weights(f"input/tps2203-lstm-output/model.tf")


def make_test_dataset(df, congestions, sequence_length=32):
    data = df.copy()
    items = congestions[-sequence_length:len(congestions)] + [0]
    data["congestion"] = items
    dataset = tf.data.Dataset.from_tensor_slices((data))
    dataset = dataset.window(sequence_length + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(sequence_length + 1))
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(1)
    return dataset

import time
begin = time.time()
test = pd.read_csv("input/tabular-playground-series-mar-2022/test.csv")
test = feature_engineering(test)
submission = pd.read_csv("input/tabular-playground-series-mar-2022/sample_submission.csv")
first_batch = train.iloc[-sequence_length-1:-1]
df = pd.concat([first_batch[categorical_columns], test[categorical_columns]])
congestions = list(first_batch["congestion"])
for i in range(len(test)):
    ds = make_test_dataset(df.iloc[i: i+sequence_length+1], congestions, sequence_length=sequence_length)
    congestion = model.predict(ds)[0][0]
    congestions.append(congestion)
    if (i + 1) % 100 == 0:
        elaspsed_time = time.time() - begin
        estimated_time = elaspsed_time / (i + 1) * len(test)
        eta = estimated_time - elaspsed_time
        print(f"ETA: %.2f"%(eta))
submission["congestion"] = np.round(congestions[sequence_length:])
submission.to_csv("submission.csv", index=False)


