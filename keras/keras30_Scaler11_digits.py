### <<15>>

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import sklearn as sk
print(sk.__version__)
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf 
print(tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)                 # (1797, 64) (1797,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64)
#print(pd.value_counts(y))
# 다중분류

y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)
print(x.shape, y.shape) # (1797, 64) (1797, 10)

# # X데이터 확인
# x_df = pd.DataFrame(x, columns=datasets.feature_names)

# # 상위 5행 확인
# print("\nDataFrame 형태로 확인:")
# print(x_df.head())

# # 기본 통계 정보(전처리활때 필요한 정보)
# print("\n기술 통계:")
# pd.set_option('display.max_columns', None)  # 컬럼정보 전부 출력
# print(x_df.describe())

# # 데이터 타입 및 형태 확인
# print("\nDataFrame 구조:")
# print(x_df.info())

# # 결측치 확인
# print("\n결측치 확인:") # 결측치 없음
# print(x_df.isna().sum())    

# 훈련데이터/테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=4325,
    stratify=y,
)

# 훈련데이터에서 검증데이터 분할(검증데이터는 스케일링 적용되면 안되기때문에 여기서 분할)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, 
    test_size=0.2,
    random_state=5234,
    stratify=y_train,
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
original_x_val = x_val.copy()

for scaler_name, scaler in scalers:

    # 1. 스케일링 누적X
    x_train = original_x_train.copy()
    x_test = original_x_test.copy()
    x_val = original_x_val.copy()
    
    if scaler is not None:
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x_val = scaler.transform(x_val)
    
    # # 1. 스케일링 누적 O
    # if scaler is None:
    #     x_train = x_train
    #     x_test = x_test
    # else:
    #     scaler.fit(x_train)
    #     x_train = scaler.transform(x_train)
    #     x_test = scaler.transform(x_test)
        
    # 2. 모델구성
    model = Sequential()
    model.add(Dense(128, input_dim=64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
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

    ####################### mcp 세이브 파일명 만들기 #######################
    import datetime
    date = datetime.datetime.now()
    print(date)         # 2025-06-02 13:00:40.661379
    print(type(date))   # <class 'datetime.datetime'>
    date = date.strftime('%m%d_%H%M')
    print(date)         # 0602_1305
    print(type(date))   # <class 'str'>

    path = './_save/keras30_mcp/11_digit/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'                # 04d : 정수 4자리, .4f : 소수점 4자리
    # filepath 가변 (갱신때마다 저장)
    # filepath = "".join([path, 'k30_', date, '_', filename])     # 구분자를 공백("")으로 하겠다.
    # filepath 고정 (종료때만 저장)
    filepath = path + f'keras30_mcp3_{scaler_name}.hdf5'

    print(filepath)

    #exit()

    mcp = ModelCheckpoint(          # 모델+가중치 저장
        monitor = 'val_loss',
        mode = 'auto',
        save_best_only=True,
        filepath = filepath,
    )

    start_time=time.time()
    model.fit(
        x_train, y_train,
        epochs=200,
        batch_size=32,
        verbose=0,
        #validation_split=0.2,
        validation_data=(x_val, y_val),
        callbacks=[es, mcp],
    )
    end_time=time.time()
    
    #################### 속도 측정 ####################
    if gpus:
        print('GPU 있다~')
    else:
        print('GPU 없다~')
    print("걸린 시간 :", round(end_time-start_time, 2), "초")
    # GPU 있다~
    # 걸린 시간 : 25 초대
    
    # GPU 없다~
    # 걸린 시간 : 8 초대
    #################### 속도 측정 ####################
    
    # 4. 평가, 예측
    ## 4.1 평가
    results = model.evaluate(x_test, y_test)    # model이 알아서 argmax로 변환하고 평가한다.
    print('loss:', results[0])
    print('(categorical)acc:', results[1])

    ## 4.2 예측
    # acc 
    # model자체 평가이후 acc지표로 예측하기위해 테스트데이터 argmax변환
    y_pred = model.predict(x_test)              # 넘파이배열로 반환
    y_pred = np.argmax(y_pred, axis=1)          # 각 행 마다 가장 큰 값이 1인 인덱스 반환
    print(y_pred)                               # [5 7 ... 0 3 3]
    y_test_argmax = np.argmax(y_test, axis=1)   # ✅ 반복문 돌기위해서 원본 y_test 유지
    print(y_test_argmax)                        # [5 7 ... 0 3 3]
    acc = accuracy_score(y_test_argmax, y_pred)
    
    # f1-score 예측
    y_pred_f1 = model.predict(x_test)
    y_pred_f1 = np.argmax(y_pred_f1, axis=1)
    print(y_pred_f1)    
    
    # -> 멀티클래스 처리 : average 파라미터 추가 필요
    # f1_score  (y가 이진데이터가 아닐때 average 파라미터를 넣어줘야한다.(세 개 이상의 클래스 중요도를 어떻게 배분해야할지 명시가 필요하기때문에))
    # average파라미터없이 y_pred_f1 = (y_pred_f1 > 0.5).astype(int) # 이렇게 하면 안되는이유 : y_pred_f1의 0을 제외한 나머지 클래스 정보가 소실된다. (ex : 0.5 이상인 1,2는 같은 걸로 취급되버린다.)
    f1 = f1_score(y_test_argmax, y_pred_f1, average='macro')   # macro : 각 클래스별 F1-score를 계산한 뒤, 산술 평균(가중치X)
    
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print('acc :', acc)
    print("F1-Score :", f1)
    
"""
    === 현재 적용 스케일러: None ===
    acc : 0.9722222222222222
    F1-Score : 0.9709458204583742

    === 현재 적용 스케일러: MinMax ===
    acc : 0.975
    F1-Score : 0.9740696160554618

    === 현재 적용 스케일러: Standard ===
    acc : 0.9777777777777777
    F1-Score : 0.9772666733551881

    === 현재 적용 스케일러: MaxAbs ===
    acc : 0.9805555555555555
    F1-Score : 0.9804426090217042
    
    === 현재 적용 스케일러: Robust ===
    acc : 0.9777777777777777
    F1-Score : 0.9764760679784763
"""