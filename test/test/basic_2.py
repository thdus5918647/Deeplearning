from glob import glob
import os
import re
import numpy as np

files_list = glob(os.path.join('/home/inlabws/soyeon/test/dataset/basic/txt/', '*.txt'))

# 불러온 txt 파일 번호순으로 정렬
def solution(files):
    file_list = [re.split('([0-9]+)', i) for i in files_list]
    file_list.sort(key = lambda x : (x[0].lower(), int(x[1])))

    return [''.join(lst) for lst in file_list]

result = solution('/home/inlabws/soyeon/test/dataset/basic/txt') # 정렬된 파일

hz_data = []
amp_data = []
#amp_data_index = 0
#hz_data_index = 0

for idx in range(0, len(result)):
#for idx in range(0, 2):
    dataset = [] # "hz:amp"로 구성된 txt파일 
    hz = [] # hz데이터만 모아둔 리스트
    amp = [] # amp데이터만 모아둔 리스트

    # 번호순으로 파일 한 줄씩 읽기
    F = open("./dataset/basic/txt/basic_%d.txt"%idx, 'r').read().split('\n') 

    # hz와 amp를 따로 추출하기 위해
    # ':'을 기준으로 나누어 읽고 공백 제거 후 dataset 리스트에 삽입
    for i in range(0, len(F)):
        data_list = F[i].split(":")
        data_list = ' '.join(data_list).split() # 리스트 공백제거
        dataset.append(data_list)

    # 맨마지막줄이 공백이므로 -1만큼 반복해서 hz와 amp 리스트에 각각 데이터 삽입
    for i in range(0, len(dataset)-1):
        hz.append(dataset[i][0])
        amp.append(dataset[i][1])

    # hz 데이터 샘플링
    # 50의 간격을 두고 5개의 데이터 평균 추출
    hz_result = np.array([])
    value = 0

    for i in range(0, len(hz)):
        if i % 50 < 5:
            value += float(hz[i])
            if i % 50 == 4:
                value /= 5.0
                hz_result = np.append(hz_result, value)
                value = 0
    
    # 샘플링된 데이터들을 hz_data 리스트에 삽입
    hz_data.append(hz_result)
    #hz_data.append([])
    #hz_data[hz_data_index].append(hz_result)
    #hz_data_index += 1

    # amp 데이터 샘플링
    amp_result = np.array([])
    value = 0

    for i in range(0, len(amp)):
        if i % 50 < 5:
            value += float(amp[i])
            if i% 50 == 4:
                value /= 5.0
                amp_result = np.append(amp_result, value)
                value = 0
                
    amp_data.append(amp_result)

y = [[0 for j in range(1)] for i in range(1000)]

print(len(hz_data))
print(len(amp_data))
print(len(hz_data[0]))
print(len(y))

