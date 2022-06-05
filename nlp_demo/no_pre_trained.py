import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# 定义文档
documents = [
    'Well done!',
    'Good work',
    'Great effort',
    'nice work',
    'Excellent!',
    'Weak',
    'Poor effort',
    'not good',
    'poor work',
    'Could have done better.',
]

# 定义标签
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

vocab_size = 50
encodeDocuments = [keras.preprocessing.text.one_hot(doc, vocab_size) for doc in documents]
print(encodeDocuments)

max_length = 4
paddedDocuments = keras.preprocessing.sequence.pad_sequences(encodeDocuments, maxlen=max_length, padding='post')
print(paddedDocuments)

model = Sequential()
model.add(layers.Embedding(vocab_size, 8, input_length=max_length))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())


model.fit(paddedDocuments, labels, epochs=50, verbose=1)

loss, accuracy = model.evaluate(paddedDocuments, labels, verbose=0)
print('Accuracy：%f' % accuracy*100)







