import keras
import pymysql
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras import regularizers
import csv # CSV 파일을 읽기 위해서는 먼저 파이썬에 기본 내장된 csv 모듈을 import
import random

csv_file_name = "log-Copy1.csv" # csv 파일명, 문자열은 ""로 감싸져있어야함.
colum_title = [] # 열 제목 저장을 위한 배열
colum_data = [] # 열 데이터 저장을 위한 배열
csv_delimiter=' ' # 데이터 구분자

# python에서 파일처리를 할 때 with as를 사용하면 파일을 열고 구문이 끝나면 인터프리터가 자동으로 닫음
with open(csv_file_name) as csvfile: # csv파일 open -> csvfile로 사용
    reader = csv.reader(csvfile, delimiter=csv_delimiter, quoting=csv.QUOTE_NONNUMERIC) # csvfile을 ' ' 으로 데이터를 구분하고 숫자가 아닌 모든 필드를 읽음
    for line in reader: # reader 길이만큼 반복(line은 데이터 한 줄)
        for line_data in line: # line 길이만큼 반복
            if isinstance(line_data, str): # line_data가 string 타입인 경우
                if line_data not in colum_title: # colum_title이 line_data가 아닌 경우
                    colum_title.append(line_data) # colum_title에 line_data 추가
            else: # line_data가 string 타입이 아닌 경우
                colum_data.append(float(line_data)) # colum_data에 line_data 추가

# reshape는 배열의 내부 데이터는 보존한 채로 형태를 변형
data = np.array(colum_data).reshape(int(len(colum_data)/3),3) # 1차원 배열의 colum_data를 2차원 배열(행 : colum_data의 길이, 열 : 3)로 변경

# 로지스틱 회귀

# 실행할 때마다 같은 결과를 출력하기 위한 seed 값
seed = 0 # 0 = 섞지 않음, 0보다 크면 섞음
np.random.seed(seed) # 데이터 섞기
tf.compat.v1.set_random_seed(seed) # 데이터 섞기

# x 데이터 값
x1_data = [x1[0] for x1 in data] # dBm 데이터 = x1 데이터(data 길이만큼 반복하여 0번째 원소만)
x2_data = [x2[1] for x2 in data] # bps 데이터 = x2 데이터(data 길이만큼 반복하여 1번째 원소만)
len_x = len(x1_data) # x_data의 길이
x_data = [0]*len_x # len_x 만큼 배열 초기화
for i in range(0, len_x): # len_x 만큼 반복
    x_data[i] = [x1_data[i], x2_data[i]] # x_data에 dBm과 bps 데이터를 넣음
x_data = np.array(x_data)

print(x_data)

# y 데이터 값
y_data = [y_row[2] for y_row in data] # y 데이터(data 길이만큼 반복하여 2번째 원소만)
# 조건으로 y 데이터 값을 정리
for i in range(0, len_x): # len_x 만큼 반복
    if x1_data[i] > -80 and x2_data[i] > 1000: # dBm이 -80보다 크고 bps가 1000보다 큰 경우
        y_data[i] = 1 # 해당 y 데이터 값을 1로 변경

y_data = np.array(y_data).reshape(len_x, 1) # 2차원 배열 선언(행 : len_x, 열 : 1)

print(y_data)
(train_x,test_x,train_y,test_y)= train_test_split(x_data,y_data,test_size=0.3)

from tensorflow.compat.v2.keras.models import model_from_json

# model.json 파일 열기
json_file = open("keras_lora-Copy1.json", "r")
loaded_model_json = json_file.read()
json_file.close()

# json파일로 부터 model 로드하기
loaded_model = model_from_json(loaded_model_json)

loaded_model.summary()
# 로드한 model에weight 로드하기
loaded_model.load_weights("keras_lora-Copy1.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])

loaded_model.fit(train_x, train_y, epochs=500, batch_size=500,validation_data=(test_x,test_y),verbose=0)

score=loaded_model.evaluate(test_x,test_y,verbose=0)

print(len(train_x), "train +", len(test_x), "test")
print("\n Loss: %.4f. Accuracy: %.4f" % (loaded_model.evaluate(x_data, y_data)[0],loaded_model.evaluate(x_data, y_data)[1]))
print("\n Test loss: ", score[0],"\n Test accuracy: ", score[1])

'''
x_input = np.array([-47,4568])
x_input = x_input.reshape((1,2))
y = loaded_model.predict(x_input, verbose=0)
print(y)
'''

while True:
    try:
        x1 = int(input("dBm : "))
        x2 = float(input("bps : "))
        break
    except:
        print("다시 입력해주세요!")

new_x = np.array([x1, x2]).reshape(1,2)
new_y = loaded_model.predict(new_x, verbose=0)
print("dBm : %d, bps : %f" % (new_x[:,0], new_x[:,1]))
print("확률 : %6.2f %%" %(new_y*100))
