import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

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
# 학습 데이터와 테스트 데이터를8:2로 나누고, 데이터의 순서 또한 섞음
(X_train, X_test, y_train, y_test) = train_test_split(data_X, data_y, train_size=0.8, random_state = 1)

print(len(X_train))
print(X_train.shape)

# 학습 데이터와 테스트 데이터에 대해 원-핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

model.add(Dense(64, activation='relu', input_dim=4))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
variables_for_classification=5
model.add(Dense(variables_for_classification, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size = 128)

score = model.evaluate(X_test, y_test, batch_size = 128)
