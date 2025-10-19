### <<18>>

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


# 원핫인코딩
# y reshape (1차원으로)
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(128, (3,3), strides=1, input_shape=(32, 32, 3), activation='relu'))                      
model.add(MaxPool2D())
model.add(Dropout(0.3))                                 
model.add(Conv2D(64, (3,3), activation='relu'))        
model.add(MaxPool2D())                                  
model.add(Dropout(0.2))              
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())    
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=16))
model.add(Dense(units=10, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'],
              )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', 
                   mode='min',
                   patience=100,
                   verbose=1,   # stop이 어느 epoch에서 걸렸는지 출력해줌(modelcheckpoint도 적용가능)
                   restore_best_weights=True,
                   )

####################### mcp 세이브 파일명 만들기 #######################
# import datetime
# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')

path = './_save/keras39/'
filename = '.hdf5'             
filepath = "".join([path, 'k39_cifar10', filename])     # 구분자를 공백("")으로 하겠다.

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
                 epochs=400, 
                 batch_size=128, 
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
    loss : 0.8048772215843201
    accuracy :  0.7418
    걸린 시간 : 502.58 초
    
    loss : 0.7644357681274414
    accuracy :  0.7405
    걸린 시간 : 683.58 초
    
    loss : 0.7444074153900146
    accuracy :  0.7518
    걸린 시간 : 800.18 초
    
    loss : 0.74698805809021
    accuracy :  0.7576    
    걸린 시간 : 1062.74 초
    
"""
