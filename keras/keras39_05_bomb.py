### <<18>>

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
#        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
#        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))

# 시각적으로확인
import matplotlib.pyplot as plt
# 100개 클래스 시각화 (10x10 그리드)
plt.figure(figsize=(25, 25))
for i in range(100):  # 0~99 클래스 반복
    class_indices = np.where(y_train.reshape(-1) == i)[0]  # 1차원 변환 후 인덱스 탐색
    if len(class_indices) > 0:  # 해당 클래스가 존재하는 경우
        plt.subplot(10, 10, i+1)  # 10행 10열 서브플롯
        plt.imshow(x_train[class_indices[0]])  # 각 클래스 첫 번째 이미지
        plt.title(f'Class {i}', fontsize=8)
        plt.axis('off')
plt.tight_layout()
#plt.show()

# 정규화
x_train = x_train/255.
x_test = x_test/255.
print(x_train.shape, x_test.shape)
print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0


# 원핫인코딩
# y reshape (1차원으로)
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)


# 2. 모델구성
model = Sequential()

# 특징 추출기(Feature Extractor) 확장
model.add(Conv2D(1024, (3,3), strides=1, input_shape=(32,32,3), activation='relu', padding='same'))  # [1]
model.add(BatchNormalization())  # 배치 정규화 추가
model.add(Conv2D(1024, (3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))  # 드롭아웃 강화

model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())  # Flatten 대체

# 분류기(Classifier) 확장
model.add(Dense(1024, activation='relu', kernel_regularizer='l2'))  # L2 정규화 추가
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(100, activation='softmax'))  # 출력층 100개 유닛

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'],
              )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_acc', 
                   mode='max',
                   patience=30,
                   verbose=1,   # stop이 어느 epoch에서 걸렸는지 출력해줌(modelcheckpoint도 적용가능)
                   restore_best_weights=True,
                   )

####################### mcp 세이브 파일명 만들기 #######################
# import datetime
# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')

path = './_save/keras39/'
filename = '.hdf5'             
filepath = "".join([path, 'k39_cifar100', filename])     # 구분자를 공백("")으로 하겠다.

print(filepath)

mcp = ModelCheckpoint(          # 모델+가중치 저장
    monitor = 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath,
)
start = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=2000, 
                 batch_size=1024, 
                 verbose=1, 
                 validation_split=0.2,
                 callbacks=[es, mcp],
                 )
end = time.time()

## 그래프 그리기
plt.figure(figsize=(18, 5))
# 첫 번째 그래프
plt.subplot(1, 2, 1)  # (행, 열, 위치)
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('bank Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid()

# 두 번째 그래프
plt.subplot(1, 2, 2)
plt.plot(hist.history['acc'], c='green', label='acc')
plt.plot(hist.history['val_acc'], c='orange', label='val_acc')
plt.title('bank Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.grid()

plt.tight_layout()  # 간격 자동 조정
plt.show()  # 윈도우띄워주고 작동을 정지시킨다. 다음단계 계속 수행하려면 뒤로빼던지

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)   # evaluation도 verbose 옵션사용가능
print('loss :', loss[0])
print('acc :', loss[1])

y_pred = model.predict(x_test)
print(y_pred.shape) # (10000, 10)
print(y_test.shape) # (10000, 10)

# argmax이용하여 최대값있는 인덱스만 벡터(시리즈)로 반환후 얼마나 일치하는지 계산
y_pred = np.argmax(y_pred, axis=1)      # axis = 1 : 행 방향
print(y_pred)           # [1 6 1 ... 1 1 6]
y_test = y_test.values  # nparray로 변환. values : 속성. 값만 반환
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_pred)
print("accuracy : ", acc)
print("걸린 시간 :", round(end-start, 2), "초")

"""
    loss : 2.5189974308013916
    accuracy :  0.351
    걸린 시간 : 975.77 초
    
"""
