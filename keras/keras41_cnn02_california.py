### <<19>>

# 30_2 카피
# # SSL 비활성화 (한번만 실행하면됨)
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import sklearn as sk 
print(sk.__version__)
import tensorflow as tf 
print(tf.__version__)
import numpy as np

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, BatchNormalization, MaxPool2D
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
    x_train = x_train.reshape(-1, 8, 1, 1)
    x_test  = x_test.reshape(-1, 8, 1, 1)
    # (-1, 4, 2, 1) 같은 형태로 reshape하지 않는 이유 : Conv2D가 의미없는 공간정보를 학습하여 성능을 떨어트릴수도있음. 원본정보 보존을 위해 (-1, 8, 1, 1) 같은 형태가 좋음 
    # 예) 연봉과 혈압이 인접해 의미 없는 상관관계를 학습할 수 있음.
    # 또한 컬럼별로 달리 적용된 스케일링 값들이 공간학습하면서 혼재되버려 의도치않은 상호작용 발생할 수있음
    
    # 2. 모델구성
    # model = Sequential()
    # model.add(Dense(128, input_dim=8))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(1, activation='linear')) 
    
    # 공간관계 학습을 차단하고 독립성을 보장하게 모델을 작성한다.
    # kernel_size(1,1) // input_shape(8,1,1) // strides=1
    """
    [Conv2D와 Dense 차이]
    표현력: Dense 레이어가 훨씬 유연합니다. 공간적 제약 없이 모든 입력 간 관계를 학습합니다.

    과적합 위험: Dense는 파라미터가 많아 과적합 가능성 ↑, Conv2D는 과소적합 가능성 ↑

    계산 효율성: Conv2D가 파라미터가 적어 빠른 학습이 가능합니다.
    
    Dense는 각 출력 뉴런이 독립적인 가중치가진다.
    """
    
    model = Sequential()
    model.add(Conv2D(128, (1,1), padding='same', strides=1, input_shape=(8, 1, 1), activation='relu')) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)        
    #model.add(MaxPool2D(pool_size=(2, 1)))
    # MaxPool2D의 디폴트 옵션 : MaxPool2D(pool_size=(2,2), strides=pool size와 동일, padding=valid)
    model.add(Dropout(0.3))                                 
    model.add(Conv2D(64, (1,1), padding='same', activation='relu'))        
    #model.add(MaxPool2D(pool_size=(2, 1)))                                  
    model.add(Dropout(0.2))              
    model.add(Conv2D(32, (1,1), padding='same', activation='relu'))
    model.add(Flatten())    
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))
    model.summary()

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
    
"""
    [dnn 방식]
    === 현재 적용 스케일러: None ===
    r2 스코어 : 0.6578669955345715

    === 현재 적용 스케일러: MinMax ===
    r2 스코어 : 0.802519580502156

    === 현재 적용 스케일러: Standard ===
    r2 스코어 : 0.7873720668774125

    === 현재 적용 스케일러: MaxAbs ===
    r2 스코어 : 0.8125831497145666

    === 현재 적용 스케일러: Robust ===
    r2 스코어 : 0.7868825721246113
    
    [cnn 방식 : kernel_size=(2,2), input_shape(4,2,1)] -> 무의미한 공간정보학습으로 성능 다소 떨어진것으로 추정됨
    === 현재 적용 스케일러: None ===
    r2 스코어 : 0.6420514920800353
    
    === 현재 적용 스케일러: MinMax ===
    r2 스코어 : 0.7972870368256908
    
    === 현재 적용 스케일러: Standard ===
    r2 스코어 : 0.8120298297450093
    
    === 현재 적용 스케일러: MaxAbs ===
    r2 스코어 : 0.6990483147567976
    
    === 현재 적용 스케일러: Robust ===
    r2 스코어 : 0.8005076163909606
    
    [cnn 방식 : kernel_size=(1,1), input_shape(8,1,1)] -> 성능이 제일 처참
    === 현재 적용 스케일러: None ===
    r2 스코어 : 0.6644170118709947
    
    === 현재 적용 스케일러: MinMax ===
    r2 스코어 : 0.6998414153794268
    
    === 현재 적용 스케일러: Standard ===
    r2 스코어 : 0.7888536825057522
    
    === 현재 적용 스케일러: MaxAbs ===
    r2 스코어 : 0.6085089994054894
    
    === 현재 적용 스케일러: Robust ===
    r2 스코어 : 0.7886363965252153
"""

