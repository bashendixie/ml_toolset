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

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(documents)
vocab_size = len(tokenizer.word_index) + 1
encodeDocuments = tokenizer.texts_to_sequences(documents)
print(encodeDocuments)

max_length = 4
paddedDocuments = keras.preprocessing.sequence.pad_sequences(encodeDocuments, maxlen=max_length, padding='post')
print(paddedDocuments)

# load glove model
inMemooryGlove = dict()
f = open('D:\deeplearn\dataset\glove.6B\glove.6B.100d.txt', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefficients = np.asarray(values[1:], dtype='float32')
    inMemooryGlove[word] = coefficients
f.close()
print(len(inMemooryGlove))

# create coefficient matrix for training data
trainingToEmbeddings = np.zeros((vocab_size, 100))
for word,i in tokenizer.word_index.items():
    gloveVector = inMemooryGlove.get(word)
    if gloveVector is not None:
        trainingToEmbeddings[i] = gloveVector


model = Sequential()
model.add(layers.Embedding(vocab_size, 100, weights=[trainingToEmbeddings], input_length=max_length, trainable=False))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

model.fit(paddedDocuments, labels, epochs=50, verbose=1)

loss, accuracy = model.evaluate(paddedDocuments, labels, verbose=0)
print('Accuracy：%f' % accuracy*100)
