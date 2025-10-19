### <<19>>

# 30-3 카피

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, BatchNormalization, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np 

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
    x_train = x_train.reshape(-1, 5, 2, 1)
    x_test  = x_test.reshape(-1, 5, 2, 1)

    # 2. 모델구성
    # model = Sequential()
    # model.add(Dense(128, input_dim=10))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(1, activation='linear')) 
    model = Sequential()
    model.add(Conv2D(128, (2,2), padding='same', strides=1, input_shape=(5, 2, 1), activation='relu')) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)        
    #model.add(MaxPool2D(pool_size=(2, 1)))
    # MaxPool2D의 디폴트 옵션 : MaxPool2D(pool_size=(2,2), strides=pool size와 동일, padding=valid)
    model.add(Dropout(0.3))                                 
    model.add(Conv2D(64, (2,2), padding='same', activation='relu'))        
    #model.add(MaxPool2D(pool_size=(2, 1)))                                  
    model.add(Dropout(0.2))              
    model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
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
        patience=100,             
        restore_best_weights=True,  
    )

    hist = model.fit(x_train, y_train, 
                    epochs=1000, 
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
    === 현재 적용 스케일러: None ===
    r2 스코어 : 0.5812792097799544

    === 현재 적용 스케일러: MinMax ===
    r2 스코어 : 0.576058329292772

    === 현재 적용 스케일러: Standard ===
    r2 스코어 : 0.5832858716476195

    === 현재 적용 스케일러: MaxAbs ===
    r2 스코어 : 0.5735189401397506

    === 현재 적용 스케일러: Robust ===
    r2 스코어 : 0.5903033244606405
"""

