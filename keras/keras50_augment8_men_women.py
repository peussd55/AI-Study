### <<22>>

# 46-8 카피

# C:\Study25\_data\kaggle\men_women

import numpy as np
import pandas as pd
import time
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 1. 데이터
np_path = 'c:/study25/_data/_save_npy/'

x_train_load = np.load(np_path + "keras46_07_x_train_size220.npy")
y_train_load = np.load(np_path + "keras46_07_y_train_size220.npy")

# 데이터불균형 확인
# # pd방법
# print(pd.value_counts(y_train_load))
# print(pd.DataFrame(y_train_load).value_counts())
# print(pd.Series(y_train_load).value_counts())
# np방법
print(np.unique(y_train_load, return_counts=True))  # (array([0., 1.], dtype=float32), array([1409, 1900], dtype=int64))

# test데이터가 따로 없으므로 train 데이터에서 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_train_load, y_train_load,
    test_size = 0.3,
    # stratify=y_train_load,
    random_state=8282,
)

print(x_train.shape, x_test.shape)  # (2316, 220, 220, 3) (993, 220, 220, 3)
print(y_train.shape, y_test.shape)  # (2316,) (993,)

# 훈련데이터 증폭(클래스가 0 인것만 증폭)
datagen = ImageDataGenerator(       # 증폭 준비
    # rescale=1./255,                 # 44-1 에서 50x50사이즈와 함께 정규화했음
    horizontal_flip=True,           # 수평 반전 <- 데이터 증폭 또는 변환 / 좌우반전
    vertical_flip=True,           # 수직 반전 <- 데이터 증폭 또는 변환 / 상하반전
    width_shift_range=0.1,          # 평행이동 10% (너비의 10% 이내범위에서 좌우 무작위 이동)
    # height_shift_range=0.1,       # 수직이동 10% (높이의 10% 이내범위에서 좌우 무작위 이동)
    rotation_range=15,              # 회전 5도
    # zoom_range=1.2,               # 확대 1.2배
    # shear_range=0.7,              # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(짜부러트리기)
    # fill_mode='nearest',            # 물체가 이동해서(잡아당겨져서) 생기게 된 공간을 근처픽셀값에 근접한 값으로 대체
)   # 다 살리면 쓰레기 이미지까지 생성(증폭)됨.

augment_size = 500      # 500개 증폭시킬것임
print(x_train.shape)    # (2316, 220, 220, 3)
print(x_train.shape[0]) # 2316

# y가 0인 클래스만 증폭시키기위해 y가 0인 것만 추출
# randidx = np.random.randint(x_train.shape[0], size=augment_size)
zero_indices = np.where(y_train == 0)[0]
randidx = np.random.choice(zero_indices, size=augment_size, replace=False)

print(randidx)                              # [441  ...  518]
print(np.min(randidx), np.max(randidx))     # 4 2313

# 데이터생성
x_augmented = x_train[randidx].copy()   # 1000개의 데이터 copy, copy로 새로운 메모리 할당. 서로영향X
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)    # (500, 220, 220, 3)
print(x_augmented.shape)    # (500, 220, 220, 3)

# # 차원변환 : 이미 4차원이라 변환 불필요
# # x_augmented = x_augmented.reshape(500, 220, 220, 3)
# x_augmented = x_augmented.reshape(
#     x_augmented.shape[0],
#     x_augmented.shape[1],
#     x_augmented.shape[2],
#     3,
#     )
# print(x_augmented.shape)    # (500, 220, 220, 3)

# 위에서 정의된 x_augmented 갯수만큼 변형시킨다.
x_augmented = datagen.flow( # 이부분에서 실제 변환(증폭)이됨. 훈련데이터만 증폭할것이기때문에 x_augmented만 datagen.flow()실행하고 뒤에서 이어붙인다.
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0] #튜플(next())에서 x만(0) 추출

print(type(x_augmented))    # <class 'numpy.ndarray'> : 위에서 [0]으로 첫번째 요소(튜플 x값) 뽑아왔음
print(x_augmented.shape)    # (500, 220, 220, 3)

# x데이터 reshape
print(x_train.shape, x_test.shape)  # (2316, 220, 220, 3) (993, 220, 220, 3)

x_train = np.concatenate([x_train, x_augmented])
y_train = np.concatenate([y_train, y_augmented])
print(x_train.shape, y_train.shape) # (2816, 220, 220, 3) (2816,)

# # y 원핫인코딩 : 이미 라벨된 .npy 불러와서 불필요
# print(y_train.shape, y_test.shape)  # (27500,) (7500,)
# # # y reshape (1차원으로)   # 이미 y_train, y_test가 1차원이라 변경 불필요
# # y_train = y_train.reshape(-1)
# # y_test = y_test.reshape(-1)
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
# print(y_train.shape, y_test.shape)  # (1018, 2) (309, 2)

# 2. 모델구성 : 레이어 unit이 너무 많으면 메모리 부족으로 학습안된다.
# 학습시 데이터 이동단계 : 학습데이터 cpu메모리에서 gpu메모리로 업로드함 -> gpu메모리(전용메모리)가 부족하면 학습불가 -> 분할배치나 생성자를 사용해야함.
model = Sequential()
model.add(Conv2D(32, (3,3), strides=1, input_shape=(220, 220, 3))) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)                    
model.add(MaxPool2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))       
                                            
model.add(Conv2D(64, (3,3), strides=1, activation='relu')) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)                    
model.add(MaxPool2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 

model.add(Conv2D(128, (3,3), strides=1, activation='relu')) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)                    
model.add(MaxPool2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 

model.add(Flatten())    
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))
# model.summary()

# path = './_save/kaggle/keras50_augment8/'
# model = load_model(path + 'k50_0617_1618_0050-0.7677.hdf5')      # restore_best_weights=True일때 저장한 모델과 가중치 동일

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
            optimizer='adam', 
            metrics=['acc'],
            )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping( 
    monitor = 'val_acc',       
    mode = 'auto',              
    patience=100,          
    verbose=1,     
    restore_best_weights=True, 
)

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path = './_save/keras50_augment8/'
# filepath 가변 (갱신때마다 저장)
filename = '{epoch:04d}-{val_acc:.4f}.hdf5'    # 04d : 정수 4자리, .4f : 소수점 4자리
filepath = "".join([path, 'k50_', date, '_', filename])     # 구분자를 공백("")으로 하겠다.
# filepath 고정 (종료때만 저장)
# filepath = path + f'keras46_mcp.hdf5'

mcp = ModelCheckpoint(          # 모델+가중치 저장
    monitor = 'val_acc',
    mode = 'auto',
    save_best_only=True,
    filepath = filepath,
    verbose=1,
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                batch_size = 64,
                epochs=500,
                verbose=3, 
                validation_split=0.2,
                callbacks=[es,
                            mcp,
                           ],
                )
end_time = time.time()

# 그래프 그리기
plt.figure(figsize=(18, 5))
# 첫 번째 그래프
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('img loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid()

# 두 번째 그래프
plt.subplot(1, 2, 2)
plt.plot(hist.history['acc'], c='red', label='acc')
plt.plot(hist.history['val_acc'], c='blue', label='val_acc')
plt.title('img acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.grid()
plt.tight_layout()  # 간격 자동 조정

# 4. 평가, 예측
# 4.1 평가
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc :', results[1])

# 4.2 예측
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, np.round(y_pred)) 
print('걸린시간 :', round(end_time-start_time, 2), '초')
acc2 = acc
print('acc(sklearn 지표) :', acc)

# 증폭X(220x220)
# 걸린시간 : 390.85 초
# acc(sklearn 지표) : 0.7150050352467271

# 증폭O
# 걸린시간 : 721.23 초
# acc(sklearn 지표) : 0.67472306143001
# 시간더걸리고 성능향상X (대신 내성별 예측은 정확해짐)

plt.show()