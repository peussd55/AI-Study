### <<16>>

# 30-2 카피

# (2) SSL 비활성화 (한번만 실행하면됨)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sklearn as sk 
print(sk.__version__)
import tensorflow as tf 
print(tf.__version__)
import numpy as np

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

#[실습]
#r2_score > 0.59

# 1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(x)    # 데이터가 실수로 되어있다. -> 회귀모델사용 (0,1 같은 데이터 : 분류모델사용)
print(y)    # 데이터가 실수로 되어있다. -> 회귀모델사용 (0,1 같은 데이터 : 분류모델사용)
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=1111
)

scalers = [
    ('None', None),
    ('MinMax', MinMaxScaler()),
    ('Standard', StandardScaler()),
    ('MaxAbs', MaxAbsScaler()),
    ('Robust', RobustScaler())
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
    model.add(Dense(128, input_dim=8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear')) 

    # 3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam')

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping( 
        monitor = 'val_loss',       
        mode = 'min',               
        patience=30,             
        restore_best_weights=True,  
    )

    hist = model.fit(x_train, y_train, 
                    epochs=300, 
                    batch_size=24, 
                    verbose=3, 
                    validation_split=0.2,
                    callbacks=[es],
                    ) 
        
    # 4. 평가, 예측
    loss = model.evaluate(x_test, y_test)
    results = model.predict(x_test)
    
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")

    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_test, results)
    print('r2 스코어 :', r2)

    # 실험결과 정리
    # 학습을돌릴때 loss가 주기적으로 튀는 경우가있다 이럴 경우의 파라미터는 좋지 않은 케이스니 소거한다.
    # model.add(Dense(512))가 추가되면 학습과정중에 loss가 주기적으로 몇만씩 튀어서 loss가 잘 안내려간다.
    # 모델 훈련(fit)때 loss는 낮게 나오는데 평가(evluate)때 loss가 높게 나오는 경우 : 과적합된경우임
    
"""
    === 현재 적용 스케일러(dropout X): None ===
    r2 스코어 : 0.6361828925868854
    
    === 현재 적용 스케일러(dropout O): None ===
    r2 스코어 : 0.0007106274387985723
    
    ==> dropout 성능 down
"""

