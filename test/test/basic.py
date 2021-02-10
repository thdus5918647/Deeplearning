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

#basic_data = [] # 샘플링 후 결과 데이터
#basic_data_index = 0

hz_data = []
hz_data_index = 0 
amp_data = []

#for idx in range(0, len(result)):
for idx in range(0, 2):
    dataset = []
    hz = []
    amp = []

    F = open("./dataset/basic/txt/basic_%d.txt"%idx, 'r').read().split('\n')
    #print(F)

    for i in range(0, len(F)):
        data_list = F[i].split(":")
        data_list = ' '.join(data_list).split() # 리스트 공백제거
        dataset.append(data_list)
    #print(dataset)
    #print(len(dataset))

    for i in range(0, len(dataset)-1):
        hz.append(dataset[i][0])
        amp.append(dataset[i][1])

    #print(hz[len(hz)-1])
    #print(amp[len(amp)-1])

    hz_result = np.array([])
    value = 0

    for i in range(0, len(hz)):
        if i % 50 <  5:
            value += float(hz[i])

            if i % 50 == 4:
                value /= 5.0
                hz_result= np.append(hz_result, value)
                value = 0

    #print(hz_result)
    #print(len(hz_result))

    amp_result = np.array([])
    value = 0

    for i in range(0, len(amp)):
        if i % 50 < 5:
            value += float(amp[i])

            if i % 50 == 4:
                value /= 5.0
                amp_result = np.append(amp_result, value)
                value = 0

    #print(amp_result)

    #basic_result = np.column_stack([hz_result, amp_result])
   
    for i in range(0, len(hz_result)):
        hz_data.append([])
        hz_data[hz_data_index].append(hz_result)
        hz_data_index += 1

    for i in range(0, len(amp_result)):
        amp_data.append(amp_result)

#print(hz_data)
print(hz_data[0])
print(len(hz_data[0]))
print('\n')
print(hz_data[0][0])
print(len(hz_data[0][0]))
#print(len(hz_data[1][0]))
print(len(hz_data))
'''
    for i in range(0, len(basic_result)):
        basic_data.append([])
        basic_data[basic_data_index].append(basic_result[i][0])
        basic_data[basic_data_index].append(basic_result[i][1])
        basic_data_index = basic_data_index + 1

    #print(idx)

#print(len(ambul_data))
print(basic_data)

#for i in range(0, len(ambul_data)):
    #print(ambul_data[i][0], ambul_data[i][1])
'''
