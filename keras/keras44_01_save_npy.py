### <<20>>

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

path_train = './_data/kaggle/cat_dog/train2/'
path_test = './_data/kaggle/cat_dog/test2/'

xy_train = train_datagen.flow_from_directory( 
    path_train,                 # 작업 경로
    target_size=[50, 50],     # 픽셀크기 일괄조정
    batch_size=100,              
    class_mode='binary',        # 이진분류
    color_mode='rgb',        
    shuffle=True,
    seed=333,                   # 시드값 고정
)
# Found 25000 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,                  # 작업 경로
    target_size=[50, 50],     # 픽셀크기 일괄조정
    batch_size=100,             
    class_mode='binary',        # 이진분류
    color_mode='rgb',        
    # shuffle=True,             # 테스트 데이터는 섞을 필요없음(default : False)
)
# Found 12500 images belonging to 2 classes.

print(xy_train[0][0].shape) # (100, 200, 200, 3)
print(xy_train[0][1].shape) # (100,)
print(len(xy_train))        # 250
end1 = time.time()
print('배치업로드 완료시간 :', round(end1-start1, 2), '초')  # 배치업로드 완료시간 : 2.07 초

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
end2 = time.time()
print('변환시간 :', round(end2-start2, 2), '초')                # 변환시간 : 43.59 초

############ 리스트를 하나의 numpy 배열로 합친다. ############
start3 = time.time()
x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)
print('x.sahpe : ', x.shape)    # (25000, 250, 250, 3)
print('y.sahpe : ', y.shape)    # (25000,)
end3 = time.time()
print('numpy 배열 합체 시간 :', round(end3-start3, 2), '초')    # numpy 배열 합체 시간 : 193.14 초
# 여기까지의 작업은 시스템 메모리 진행한다.

############ 합친 numpy를 저장한다. ############
start4 = time.time()
np_path = 'c:/study25/_data/_save_npy/'
np.save(np_path + 'keras44_01_x_train_size50.npy', arr=x)
np.save(np_path + 'keras44_01_y_train_size50.npy', arr=y)
end4 = time.time()

print('npy 저장시간 :', round(end4-start4, 2), '초')        # npy 저장시간 : 395.01 초


############### 테스트 데이터 npy 저장 #############
# 파일명: create_submission_npy_like_train.py (예시)

import numpy as np
import pandas as pd
import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 경로 및 설정 정의
path_test_dir = 'c:/study25/_data/kaggle/cat_dog/test2/test/'  # 실제 테스트 이미지가 있는 폴더
np_path = 'c:/study25/_data/_save_npy/'
npy_filename = 'keras44_01_x_submission_size50.npy'

# 2. 이미지 파일 목록으로 DataFrame 생성
# 파일 순서가 중요하므로, 운영체제가 읽어온 순서 그대로 사용합니다.
test_filenames = os.listdir(path_test_dir)
test_df = pd.DataFrame({'filename': test_filenames})

# 3. ImageDataGenerator 생성 (테스트용)
# 학습 데이터와 마찬가지로 스케일링만 적용합니다.[6]
test_datagen = ImageDataGenerator(rescale=1./255)

# 4. flow_from_dataframe으로 제너레이터 생성
print("제출용 데이터 제너레이터 생성을 시작합니다...")
start1 = time.time()
submission_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=path_test_dir,
    x_col='filename',
    y_col=None,                # 레이블이 없으므로 None
    target_size=(50, 50),
    batch_size=100,            # train 데이터 처리 시와 동일한 배치 사이즈
    class_mode=None,           # 레이블이 없으므로 None
    shuffle=False              # ★★★ 순서 유지를 위해 반드시 False ★★★
)
end1 = time.time()
print(f"제너레이터 생성 완료. 걸린 시간: {round(end1-start1, 2)}초")

# 5. 모든 배치 데이터를 리스트에 추가 (train 데이터 처리 방식과 동일)
print("배치 데이터를 리스트에 추가하는 작업을 시작합니다...")
start2 = time.time()
all_x_submission = []
for i in range(len(submission_generator)):
    # class_mode=None 이므로, 제너레이터는 y 없이 x_batch만 반환합니다.
    x_batch = submission_generator[i]
    all_x_submission.append(x_batch)
end2 = time.time()
print(f"리스트 변환 시간: {round(end2-start2, 2)}초")

# 6. 리스트를 하나의 numpy 배열로 합치기
print("Numpy 배열 합치는 작업을 시작합니다...")
start3 = time.time()
x_submission = np.concatenate(all_x_submission, axis=0)
end3 = time.time()
print(f"생성된 제출용 데이터 shape: {x_submission.shape}")
print(f"Numpy 배열 합체 시간: {round(end3-start3, 2)}초")

# 7. 합친 numpy 배열을 .npy 파일로 저장
print(".npy 파일 저장을 시작합니다...")
start4 = time.time()
if not os.path.exists(np_path):
    os.makedirs(np_path)
np.save(os.path.join(np_path, npy_filename), arr=x_submission)
end4 = time.time()
print(f".npy 저장 시간: {round(end4-start4, 2)}초")
print(f"제출용 데이터가 성공적으로 '{os.path.join(np_path, npy_filename)}'에 저장되었습니다.")
