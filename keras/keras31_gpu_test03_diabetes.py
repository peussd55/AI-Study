### <<15>>

# 30-3 카피

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

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np 
import time

#[실습]
#r2_score > 0.62

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=1111
)

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
    model.add(Dense(128, input_dim=10))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear')) 

    # 3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam')

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping( 
        monitor = 'val_loss',       
        mode = 'min',               
        patience=100,             
        restore_best_weights=True,  
    )
    start_time = time.time()
    hist = model.fit(x_train, y_train, 
                    epochs=200, 
                    batch_size=32, 
                    verbose=3, 
                    validation_split=0.2,
                    #callbacks=[es],
                    )
    end_time = time.time()
    
    #################### 속도 측정 ####################
    if gpus:
        print('GPU 있다~')
    else:
        print('GPU 없다~')
    print("걸린 시간 :", round(end_time-start_time, 2), "초")
    # GPU 있다~
    # 걸린 시간 : 8.89 초
    
    # GPU 없다~
    # 걸린 시간 : 4.33 초
    #################### 속도 측정 ####################

    # 4. 평가, 예측
    loss = model.evaluate(x_test, y_test)
    results = model.predict(x_test)
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_test, results)
    print('r2 스코어 :', r2)


