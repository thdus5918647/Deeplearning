import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, GlobalAveragePooling1D, Bidirectional
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np

vocab_size = 10000
embedding_dim = 64


data = pd.read_csv('data.csv', encoding='utf-8')
print(data.dtypes)
print(len(data))
print(data[:5])

data_X = data[['constraint','mobility','capability','context']].values
data_y = data['y'].values

print(data_X[:5])
print(data_y[:5])

print(type(data_X))
print(type(data_y))

print(data_X.shape)
print(data_y.shape)

data_X = data_X.tolist()

print(type(data_X))


tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_X)
print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(data_X))

data_X = tokenizer.texts_to_sequences(data_X)
(X_train, X_test, y_train, y_test) = train_test_split(data_X, data_y, train_size = 0.8, random_state = 1)

#model = tf.keras.Sequential([
#    tf.keras.layers.Embedding(vocab_size, embedding_dim),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
#    tf.keras.layers.Dense(embedding_dim, activation='relu'),
#    tf.keras.layers.Dropout(0.5),
#    tf.keras.layers.Dense(6, activation='softmax')
#    ])
"""
from tensorflow.compat.v2.keras.models import model_from_json

json_file = open("keras_lora.json","r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.summary()

loaded_model.load_weights("keras_lora.h5")
print("Loaded model from disk")

loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
loaded_model.fit(np.array(X_train), y_train, epochs=50,  validation_data=(np.array(X_test), y_test), verbose=2)
"""

model = Sequential()

model.add(Dense(7, input_dim=4, activation='tanh'))
model.add(Dense(4, activation='relu'))
model.add(Dense(7, activation='tanh'))
model.add(Dense(4, activation='relu'))
model.add(Dense(7, activation='tanh'))
model.add(Dense(4, activation='relu'))
model.add(Dense(7, activation='tanh'))
model.add(Dense(4, activation='relu'))
model.add(Dense(7, activation='tanh'))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 50

model.fit(np.array(X_train), y_train, epochs=num_epochs, validation_data=(np.array(X_test), y_test), verbose=2)
