### <<19>>

# CNN -> DNN

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

# 시각적으로확인
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 6))
for i in range(10):  # 0~9 클래스 반복
    class_idx = np.where(y_train[:, 0] == i)[0][0]  # 각 클래스의 첫 번째 인덱스 찾기
    plt.subplot(2, 5, i+1)  # 2행 5열 서브플롯
    plt.imshow(x_train[class_idx])  # 'gray' 제거(컬러 이미지 표시)
    plt.title(f'Class {i} : {y_train[class_idx][0]}')
    plt.axis('off')
plt.tight_layout()
#plt.show()

# 정규화
x_train = x_train/255.
x_test = x_test/255.
print(x_train.shape, x_test.shape)
print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

# y 원핫인코딩(판다스방식)
# y reshape (1차원으로)
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)


# 2. 모델구성
model = Sequential()
# model.add(Dense(512, input_dim=3072, activation='relu'))    # 컬럼수 : 3072
model.add(Dense(512, activation='relu', input_shape=(3072,)))
# input_shape=(x,) 이건 Dense레이어에서만 허용한다. Conv2D레이어는 input_shape 사용할때 반드시 3개(높이,너비,채널)의 인풋을 받아야한다.(흑백데이터면 input_shape=(x,y,1) 같은 방식으로)
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'],
              )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_acc', 
                   mode='max',
                   patience=50,
                   verbose=1,   # stop이 어느 epoch에서 걸렸는지 출력해줌(modelcheckpoint도 적용가능)
                   restore_best_weights=True,
                   )

# ####################### mcp 세이브 파일명 만들기 #######################
# # import datetime
# # date = datetime.datetime.now()
# # date = date.strftime('%m%d_%H%M')

# path = './_save/keras39/'
# filename = '.hdf5'             
# filepath = "".join([path, 'k39_cifar10', filename])     # 구분자를 공백("")으로 하겠다.

# print(filepath)

# mcp = ModelCheckpoint(          # 모델+가중치 저장
#     monitor = 'val_loss',
#     mode = 'auto',
#     verbose=1,
#     save_best_only=True,
#     filepath = filepath,
# )
start = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=200, 
                 batch_size=64, 
                 verbose=1, 
                 validation_split=0.2,
                 callbacks=[es],
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
plt.show()

"""
    [cnn방식]
    accuracy :  0.7418
    걸린 시간 : 502.58 초
    
    accuracy :  0.4431
    걸린 시간 : 827.63 초(cpu)
    
    accuracy :  0.4328
    걸린 시간 : 350.81 초(gpu)
    
    (정리1) : 레이어를 복잡하게해서 cnn방식보다 훈련시간을 늘려도 성능은 더 낮게나옴. (행의 수는 영향이 거의X)
    => 컬럼수 ∝ (cnn<->dnn)성능차이 
    why? - 컬럼 하나하나가 지역정보(픽셀 간 이웃 관계)를 담고있는 특징인데 cnn을 쓰면 포착할 수 있지만 dnn을 쓰면 그 정보들이 한 차원에 섞여서 포착할 수 있는 능력이 소실되기때문이다.
        -> dnn방식에서 컬럼을 지워보면 성능이 다소 향상되는데 이유는? 특징정보가 없는(대부분 0인)값으로 채워진 컬럼들이 사라지기때문
    
    (정리2) : dnn방식이라도 레이어가 복잡해질수록 gpu가 cpu보다 속도가 빨라진다.
    
"""
