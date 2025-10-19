### <<19>>

# 30_1 카피

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import sklearn as sk
print(sk.__version__)
from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, BatchNormalization, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

# 1. 데이터
datasets = load_boston()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target 
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=4325,
)

scalers = [
    ('None', None),
    ('MinMax', MinMaxScaler()),
    ('Standard', StandardScaler()),
    ('MaxAbs', MaxAbsScaler()),
    ('Robust', RobustScaler())
]

# 원본 데이터 백업
original_x_train = x_train.copy()
original_x_test = x_test.copy()

for scaler_name, scaler in scalers:
    # 1. 스케일링 누적X (차원변경됐기때문에 반드시 원본데이터가져와야함)
    x_train = original_x_train.copy()
    x_test = original_x_test.copy()
    
    if scaler is not None:
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    print(np.min(x_train), np.max(x_train))
    print(np.min(x_test), np.max(x_test))  
    
    # 4차원 reshape
    x_train = x_train.reshape(-1, 13, 1, 1)
    x_test  = x_test.reshape(-1, 13, 1, 1)

    # 2. 모델구성
    # model = Sequential()
    # model.add(Dense(10, input_dim=13))
    # model.add(Dense(11))
    # model.add(Dense(12))
    # model.add(Dense(13))
    # model.add(Dense(1))
    # model.summary()
    
    model = Sequential()
    model.add(Conv2D(128, (2,2), padding='same', strides=1, input_shape=(13, 1, 1), activation='relu')) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)        
    #model.add(MaxPool2D(pool_size=(2, 1)))
    # MaxPool2D의 디폴트 옵션 : MaxPool2D(pool_size=(2,2), strides=pool size와 동일, padding=valid)
    model.add(Dropout(0.3))                                 
    model.add(Conv2D(64, (2,2), padding='same', activation='relu'))        
    #model.add(MaxPool2D(pool_size=(2, 1)))                                  
    model.add(Dropout(0.2))              
    model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
    model.add(Flatten())    
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=16))
    model.add(Dense(units=1))
    model.summary()
    """
    _________________________________________________________________
    Layer (type)                Output Shape              Param #
    =================================================================
    conv2d_9 (Conv2D)           (None, 13, 1, 128)        640

    dropout_6 (Dropout)         (None, 13, 1, 128)        0

    conv2d_10 (Conv2D)          (None, 13, 1, 64)         32832

    dropout_7 (Dropout)         (None, 13, 1, 64)         0

    conv2d_11 (Conv2D)          (None, 13, 1, 32)         8224

    flatten_3 (Flatten)         (None, 416)               0

    dense_9 (Dense)             (None, 16)                6672

    dense_10 (Dense)            (None, 16)                272

    dense_11 (Dense)            (None, 1)                 17

    =================================================================
    """
    # 3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam')


    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping( 
        monitor = 'val_loss',      
        mode = 'min',               
        patience=100,               
        verbose=1,
        restore_best_weights=True,  
    )

    hist = model.fit(x_train, y_train, 
                    epochs=1000, 
                    batch_size=32, 
                    verbose=0, 
                    validation_split=0.1,
                    callbacks=[es],
                    )

    # 4. 평가, 예측
    loss = model.evaluate(x_test, y_test)   # 훈련이 끝난 모델의 loss를 한번 계산해서  반환
    results = model.predict(x_test)
    
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    #print('x_test :', x_test)
    #print('x_test의 예측값 :', results)
    print('loss :', loss)

    from sklearn.metrics import r2_score, mean_squared_error

    def RMSE(y_test, y_predict):
        # mean_squared_error : mse를 계산해주는 함수
        return np.sqrt(mean_squared_error(y_test, y_predict))

    rmse = RMSE(y_test, results)
    print('RMSE :', rmse)

    r2 = r2_score(y_test, results)
    print('r2 스코어 :', r2)

"""
    [dnn방식]
    === 현재 적용 스케일러: None ===
    RMSE : 4.454412767494923
    r2 스코어 : 0.7719259970039953

    === 현재 적용 스케일러: MinMax ===
    RMSE : 4.402858864048534
    r2 스코어 : 0.7771747530241184

    === 현재 적용 스케일러: Standard ===
    RMSE : 4.514431751836475
    r2 스코어 : 0.7657384287090064

    === 현재 적용 스케일러: MaxAbs ===
    RMSE : 4.4363546247043
    r2 스코어 : 0.7737714678937246

    === 현재 적용 스케일러: Robust ===
    RMSE : 4.559295978382517
    r2 스코어 : 0.7610591297097993
    
    [cnn방식]

"""