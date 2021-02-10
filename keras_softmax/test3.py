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
print(data_X[:5])
"""
vocab = sum(data_X, [])
print(vocab)

vocab_sorted = sorted(set(vocab))
print(vocab_sorted)

word_to_index = {word : index +1 for index, word in enumerate(vocab_sorted)}
print(word_to_index)

"""

(X_train, X_test, y_train, y_test) = train_test_split(data_X, data_y, train_size = 0.8, random_state = 1)

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

#model = Sequential()
#model.add(Dense(4, input_dim=4, activation='softmax'))
#model.add(Embedding(1000, 120))
#model.add(LSTM(120))
#model.add(Dense(5,activation='softmax'))

#model.add(Dense(64, activation='relu', input_dim=4))
#model.add(Dropout(0.5))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(5, activation='softmax'))

#model.add(Embedding(7200, 16, input_shape=(None,)))
#model.add(GlobalAveragePooling1D())
#model.add(Dense(16, activation='relu'))
#model.add(Dense(4, activation='softmax'))

#model.add(Dense(64, activation='relu', input_dim=4))
#model.add(Dense(48, activation='relu'))
#model.add(Dense(24, activation='relu'))
#model.add(Dense(12, activation='relu'))
#model.add(Dense(5, activation='softmax'))

#model.add(Embedding(5000, 64))
#model.add(Dropout(0.5))
#model.add(Bidirectional(LSTM(64)))
#model.add(Dense(1, activation='softmax'))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation='softmax')
    ])

model.summary()
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
#mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_baet_only=True)

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#opt = Adam(lr=0.01, decay=1e-6)

#model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'],)
#model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 50
#num_epochs = 10
#history = model.fit(np.array(X_train), y_train, epochs=num_epochs, validation_data=(np.array(X_test), y_test), verbose=2)
#model.fit(np.array(X_train), y_train, batch_size=10000, epochs=50, callbacks=[es,mc], validation_data=(np.array(X_test), y_test))

model.fit(np.array(X_train), y_train, epochs=num_epochs, validation_data=(np.array(X_test), y_test), verbose=2)
#model.fit(np.array(X_train), y_train, epochs=num_epochs, validation_data=(np.array(X_test), y_test), verbose=2)
