### <<21>>

import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

start1 = time.time()
train_datagen = ImageDataGenerator(
    rescale=1./255,             # 0 ~ 255 스케일링, 정규화
    # horizontal_flip=True,       # 수평 반전 <- 데이터 증폭 또는 변환
    # vertical_flip=True,         # 수직 반전 <- 데이터 증폭 또는 변환
    # width_shift_range=0.1,      # 평행이동 10% (너비의 10% 이내범위에서 좌우 무작위 이동)
    # height_shift_range=0.1,     # 수직이동 10% (높이의 10% 이내범위에서 좌우 무작위 이동)
    # rotation_range=5,           # 회전 5도
    # zoom_range=1.2,             # 확대 1.2배
    # shear_range=0.7,            # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(짜부러트리기)
    # fill_mode='nearest',        # 물체가 이동해서(잡아당겨져서) 생기게 된 공간을 근처픽셀값에 근접한 값으로 대체
)

test_datagen = ImageDataGenerator(  # 평가데이터는 증폭 또는 변환 하지 않는다. 형식만 맞춰주기위해 정규화한다.
    rescale=1./255,             # 0 ~ 255 스케일링, 정규화
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory( 
    path_train,                 # 작업 경로
    target_size=[200, 200],     # 픽셀크기 일괄조정
    batch_size=100,              
    class_mode='binary',        # 이진분류
    color_mode='rgb',        
    shuffle=True,
    seed=333,                   # 시드값 고정
)
# train 데이터랑 폴더구조가 같아서 같은 방식으로 저장
xy_test = test_datagen.flow_from_directory(
    path_test,                  # 작업 경로
    target_size=[200, 200],     # 픽셀크기 일괄조정
    batch_size=100,             
    class_mode='binary',        # 이진분류
    color_mode='rgb',        
    # shuffle=True,             # 테스트 데이터는 섞을 필요없음(default : False)
)

print(xy_train[0][0].shape) 
print(xy_train[0][1].shape) 
print(len(xy_train))       
end1 = time.time()
print('배치업로드 완료시간 :', round(end1-start1, 2), '초')  # 배치업로드 완료시간 : 0.14 초

print(xy_test[0][0].shape) 
print(xy_test[0][1].shape) 
print(len(xy_test))       
end1 = time.time()
print('배치업로드 완료시간 :', round(end1-start1, 2), '초')  # 배치업로드 완료시간 :  0.24 초

############ 모든 수치화된 batch데이터를 하나로 합치기 ############ 
# 모든 훈련데이터를 batch하나에 올리면 시간이 너무 오래걸리고 메모리부족으로 실패할 위험도 커서 얉게 자른 배치를 하나씩 만든다음 합치는 작업
# (100, 200, 200, 3) 를 250번 반복붙임
start2 = time.time()
all_x = []
all_y = []
for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]  # xy_train[i][0], xy_train[i][1] 을 각각 x_batch, y_batch로 할당
    all_x.append(x_batch)
    all_y.append(y_batch)
# print(all_x)

all_x_test = []
all_y_test = []
for i in range(len(xy_test)):
    x_batch_test, y_batch_test = xy_test[i]  # xy_train[i][0], xy_train[i][1] 을 각각 x_batch, y_batch로 할당
    all_x_test.append(x_batch)
    all_y_test.append(y_batch)
# print(all_x)
end2 = time.time()
print('변환시간 :', round(end2-start2, 2), '초')                # 변환시간 :  0.15 초

############ 리스트를 하나의 numpy 배열로 합친다. ############
start3 = time.time()
x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)
print('x.sahpe : ', x.shape)    # (160, 200, 200, 3)
print('y.sahpe : ', y.shape)    # (160,)

x_test = np.concatenate(all_x_test, axis=0)
y_test = np.concatenate(all_y_test, axis=0)
print('x_test.sahpe : ', x_test.shape)    # (120, 200, 200, 3)
print('y_test.sahpe : ', y_test.shape)    # (120,)
end3 = time.time()
print('numpy 배열 합체 시간 :', round(end3-start3, 2), '초')    # numpy 배열 합체 시간 : 0.02 초
# 여기까지의 작업은 시스템 메모리 진행한다.

############ 합친 numpy를 저장한다. ############
start4 = time.time()
np_path = 'c:/study25/_data/_save_npy/'
np.save(np_path + 'keras46_01_x_train.npy', arr=x)
np.save(np_path + 'keras46_01_y_train.npy', arr=y)

np.save(np_path + 'keras46_01_x_test.npy', arr=x_test)
np.save(np_path + 'keras46_01_y_test.npy', arr=y_test)
end4 = time.time()

print('npy 저장시간 :', round(end4-start4, 2), '초')        # npy 저장시간 : 0.28 초



