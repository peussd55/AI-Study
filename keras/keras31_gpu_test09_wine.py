### <<15>>

# 30-9 카피

import tensorflow as tf 
print(tf.__version__)   # 2.7.3
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터
datasets = load_wine()
x = datasets.data
y= datasets.target

print(x.shape, y.shape)
print(np.unique(y, return_counts=True)) 
#(array([0, 1, 2]), array([59, 71, 48], dtype=int64)

# print(x)
# [[1.423e+01 1.710e+00 2.430e+00 ... 1.040e+00 3.920e+00 1.065e+03]
# ...
#  [1.413e+01 4.100e+00 2.740e+00 ... 6.100e-01 1.600e+00 5.600e+02]]
# print(y)    # 다중분류 - 원핫인코딩 필요
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

# y 원핫인코딩
y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)   # nparray로 반환시키는 객체
y = ohe.fit_transform(y)
#print(y)
print(y.shape)  #(178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state = 517,
    stratify=y, # y(타겟데이터)를 계층화하겠다. y의 데이터가 적거나 불균형할때 사용. y, y_train, y_test의 각 클래스의 비율을 동등하게한다.
)
print(np.unique(y_train, return_counts=True))   # (array([0., 1.]), array([284, 142], dtype=int64)) : 원핫 인코딩을 한 상태라 0, 1만 있음
print(np.unique(y_test, return_counts=True))    # (array([0., 1.]), array([72, 36], dtype=int64)) : 원핫 인코딩을 한 상태라 0, 1만 있음

scalers = [
    ('None', None),
    # ('MinMax', MinMaxScaler()),
    # ('Standard', StandardScaler()),
    # ('MaxAbs', MaxAbsScaler()),
    # ('Robust', RobustScaler())
]
for scaler_name, scaler in scalers:
    # 데이터 스케일링 or 원본 데이터 사용
    if scaler is None:
        x_train = x_train
        x_test = x_test
    else:
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        
    # 2. 모델구성
    model = Sequential()
    model.add(Dense(32, input_dim=13, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    # 3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc'],
                )
    
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=100,
        restore_best_weights=True,
    )

    # ####################### mcp 세이브 파일명 만들기 #######################
    # import datetime
    # date = datetime.datetime.now()
    # print(date)         # 2025-06-02 13:00:40.661379
    # print(type(date))   # <class 'datetime.datetime'>
    # date = date.strftime('%m%d_%H%M')
    # print(date)         # 0602_1305
    # print(type(date))   # <class 'str'>

    # path = './_save/keras28_mcp/09_wine/'
    # filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 04d : 정수 4자리, .4f : 소수점 4자리
    # filepath = "".join([path, 'k28_', date, '_', filename])     # 구분자를 공백("")으로 하겠다.
    # # ./_save/keras27_mcp2/k27_0602_1442_{epoch:04d}-{val_loss:.4f}.hdf5

    # print(filepath)

    # #exit()

    # mcp = ModelCheckpoint(          # 모델+가중치 저장
    #     monitor = 'val_loss',
    #     mode = 'auto',
    #     save_best_only=True,
    #     filepath = filepath,
    # )

    start_time=time.time()
    model.fit(
        x_train, y_train,
        epochs=200,
        batch_size=32,
        verbose=0,
        validation_split=0.2,
        #callbacks=[es],
    )
    end_time=time.time()
    print('걸린시간:', end_time - start_time)
    
    #################### 속도 측정 ####################
    if gpus:
        print('GPU 있다~')
    else:
        print('GPU 없다~')
    print("걸린 시간 :", round(end_time-start_time, 2), "초")
    # GPU 있다~
    # 걸린 시간 : 5.77 초
    
    # GPU 없다~
    # 걸린 시간 : 3.56 초
    #################### 속도 측정 ####################
    
    # 4. 평가, 예측
    results = model.evaluate(x_test, y_test)
    print('loss:', results[0])
    print('(categorical)acc:', results[1])

    y_pred = model.predict(x_test)
    # print(y_pred)
    # [[0.32754213 0.34670812 0.32574973]
    #  ...
    #  [0.32754213 0.34670812 0.32574973]]
    # print(y_test)
    # [[0. 0. 1.]
    # ...
    #  [1. 0. 0.]]

    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)   # [1 1 1 0 2 2 0 2 1 0 1 2 2 2 1 1 0 1 1 0 0 0 1 2 2 1 0 0 1 1 2 1 0 2 2 0]
    y_test_argmax = np.argmax(y_test, axis=1)  # ✅ 원본 y_test 유지
    print(y_test_argmax)   # [1 1 0 0 2 2 0 2 1 0 1 2 2 1 1 1 0 1 1 0 0 0 1 2 2 1 0 0 1 1 2 1 0 2 2 0]

    # acc
    acc = accuracy_score(y_test_argmax, y_pred)
    
    y_pred_f1 = np.argmax(model.predict(x_test), axis=1)
    print(y_pred_f1)    
    # [1 1 1 0 2 2 0 2 1 0 1 2 2 2 1 1 0 1 1 0 0 0 1 2 2 1 0 0 1 1 2 1 0 2 2 0]

    # y_pred_f1 = (y_pred_f1 > 0.5).astype(int) # 이렇게 하면 안되는이유 : y_pred_f1의 2 클래스 정보가 소실된다. (0.5 이상인 1,2는 같은 걸로 취급되버린다.)

    # -> 멀티클래스 처리 : average 파라미터 추가 필요
    # f1_score  (y가 이진데이터가 아닐때 average 파라미터를 넣어줘야한다.(세 개 이상의 클래스 중요도를 어떻게 배분해야할지 명시가 필요하기때문에))
    f1 = f1_score(y_test_argmax, y_pred_f1, average='macro')   # macro : 동등분할
    
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print('acc :', acc)
    print("F1-Score :", f1)
    # acc : 1.0
    # F1-Score : 1.0

