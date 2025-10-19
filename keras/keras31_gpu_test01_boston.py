### <<15>>

# 30_1 카피

# gpu 활성화 옵션 OFF
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

import sklearn as sk
print(sk.__version__)
from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense 
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
    # ('MinMax', MinMaxScaler()),
    # ('Standard', StandardScaler()),
    # ('MaxAbs', MaxAbsScaler()),
    # ('Robust', RobustScaler())
]
for scaler_name, scaler in scalers:
    print(f"\n\n=== {scaler_name} Scaler 적용 ===")
    # 데이터 스케일링 or 원본 데이터 사용
    if scaler is None:
        x_train = x_train
        x_test = x_test
    else:
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    print(np.min(x_train), np.max(x_train))
    print(np.min(x_test), np.max(x_test))  

    # 2. 모델구성
    model = Sequential()
    model.add(Dense(10, input_dim=13))
    model.add(Dense(11))
    model.add(Dense(12))
    model.add(Dense(13))
    model.add(Dense(1))
    
    model.summary()

    # 3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam')


    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    # val_loss를 기준으로 최소값을 찾을건데 patience를 10으로 한다. 10번 내에 최소값을 찾으면 loss를 갱신하고 10번을 더 훈련시킨다. 10번을 훈련을 더 하는동안 최소값이 안 나타나면 훈련을 종료한다.
    es = EarlyStopping( 
        monitor = 'val_loss',       # 기준을 val_loss로
        mode = 'min',               # 최대값 : max, 알아서 찾아줘 : auto
        patience=10,               # 참는 횟수는 10번
        restore_best_weights=True,  # 가장 최소지점을 save할것인지. default = False. False가 성능이 더 잘나오면 모델이 과적합됐을 수 있음. False 마지막 종료시점의 가중치를 저장한다.
    )
    start_time = time.time()
    hist = model.fit(x_train, y_train, 
                    epochs=200, 
                    batch_size=32, 
                    verbose=3, 
                    validation_split=0.1,
                    #callbacks=[es],
                    )
    end_time = time.time()
    
    #################### 속도 측정 ####################
    if gpus:
        print('GPU 있다~')
    else:
        print('GPU 없다~')
    print("걸린 시간 :", round(end_time-start_time, 2), "초")
    # tf273gpu
    # GPU 있다~
    # 걸린 시간 : 9.02 초
    # gpu off
    # GPU 없다~
    # 걸린 시간 : 4.58 초
    #################### 속도 측정 ####################
    
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