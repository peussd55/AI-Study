### <<21>>

# C:\Study25\_data\kaggle\men_women

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
# Found 3309 images belonging to 2 classes.

test_datagen = ImageDataGenerator(  # 평가데이터는 증폭 또는 변환 하지 않는다. 형식만 맞춰주기위해 정규화한다.
    rescale=1./255,             # 0 ~ 255 스케일링, 정규화
)

path_train = './_data/kaggle/men_women/'

xy_train = train_datagen.flow_from_directory( 
    path_train,                 # 작업 경로
    target_size=[200, 200],     # 픽셀크기 일괄조정
    batch_size=100,              
    class_mode='binary',        # 다중분류
    color_mode='rgb',        
    shuffle=True,
    seed=333,                   # 시드값 고정
)

print(xy_train[0][0].shape)     # (100, 200, 200, 3)
print(xy_train[0][1].shape)     # (100, 3)
print(len(xy_train))            # 34
end1 = time.time()
print('배치업로드 완료시간 :', round(end1-start1, 2), '초')  # 배치업로드 완료시간 : 1.69 초


############ 모든 수치화된 batch데이터를 하나로 합치기 ############ 
# 모든 훈련데이터를 batch하나에 올리면 시간이 너무 오래걸리고 메모리부족으로 실패할 위험도 커서 얉게 자른 배치를 하나씩 만든다음 합치는 작업

start2 = time.time()
all_x = []
all_y = []
for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]  # xy_train[i][0], xy_train[i][1] 을 각각 x_batch, y_batch로 할당
    all_x.append(x_batch)
    all_y.append(y_batch)
# print(all_x)

############ 리스트를 하나의 numpy 배열로 합친다. ############
start3 = time.time()
x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)
print('x.sahpe : ', x.shape)    # (3309, 200, 200, 3)
print('y.sahpe : ', y.shape)    # (3309,)

# 여기까지의 작업은 시스템 메모리 진행한다.

############ 합친 numpy를 저장한다. ############
start4 = time.time()
np_path = 'c:/study25/_data/_save_npy/'
np.save(np_path + 'keras46_07_x_train_size.npy', arr=x)
np.save(np_path + 'keras46_07_y_train_size.npy', arr=y)

end4 = time.time()

print('npy 저장시간 :', round(end4-start4, 2), '초')        # npy 저장시간 :  4.48 초
