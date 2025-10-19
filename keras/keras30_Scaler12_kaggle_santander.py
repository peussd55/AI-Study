### <<15>>

# https://www.kaggle.com/competitions/santander-customer-transaction-prediction
# 이진분류

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import sklearn as sk
print(sk.__version__)
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout, BatchNormalization
import tensorflow as tf 
print(tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

# 1. 데이터
path = './_data/kaggle/santander/'           

# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.shape, test_csv.shape, submission_csv.shape)  # (200000, 202) (200000, 201) (200000, 2)
print(train_csv.head())
print(test_csv.head())

print(train_csv.columns)
# Index(['ID_code', 'target', 'var_0', 'var_1', 'var_2', 'var_3', 'var_4',
#        'var_5', 'var_6', 'var_7',
#        ...
#        'var_190', 'var_191', 'var_192', 'var_193', 'var_194', 'var_195',
#        'var_196', 'var_197', 'var_198', 'var_199'],
#       dtype='object', length=202)
print(test_csv.columns)
# Index(['ID_code', 'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5',
#        'var_6', 'var_7', 'var_8',
#        ...
#        'var_190', 'var_191', 'var_192', 'var_193', 'var_194', 'var_195',
#        'var_196', 'var_197', 'var_198', 'var_199'],
#       dtype='object', length=201)

# 결측치 확인
print(train_csv.info())
print(train_csv.isna().sum())       # 결측치 없음
print(test_csv.info())
print(test_csv.isna().sum())       # 결측치 없음

print(train_csv['target'].value_counts())
# 이진분류 (불균형)
# 0    179902
# 1     20098

# train data x로 y 분리
x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']

# 훈련데이터와 테스트데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    random_state=8282,
    shuffle=True,
    stratify=y,
)

# 훈련데이터에서 검증데이터 분할(검증데이터는 스케일링 적용되면 안되기때문에 여기서 분할)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, 
    test_size=0.3,
    random_state=5234,
    stratify=y_train,
    )

scalers = [
    ('None', None),
    ('MinMax', MinMaxScaler()),
    ('Standard', StandardScaler()),
    ('MaxAbs', MaxAbsScaler()),
    ('Robust', RobustScaler()),
]

# 원본 데이터 백업
original_x_train = x_train.copy()
original_x_test = x_test.copy()
original_x_val = x_val.copy()
original_test_csv = test_csv.copy()

# 클래스 가중치 계산 (루프 밖에서 한 번만)
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("클래스 가중치:", class_weight_dict)

for scaler_name, scaler in scalers:
    
    # 1. 스케일링 누적X
    x_train = original_x_train.copy()
    x_test = original_x_test.copy()
    x_val= original_x_val.copy()
    test_csv = original_test_csv.copy()
    
    if scaler is not None:
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x_val = scaler.transform(x_val)
        test_csv = scaler.transform(test_csv)
    # transform은 nparray를 반환(인덱스 컬럼값은 삭제됨). 할당받는 dataframe의 내부 값은 nparray이므로 할당 가능.
    # 만약 test_csv와 인덱스 순서와 submission_csv의 인덱스 순서가 다르면 인덱스순서를 맞추는 별도 작업 필요.
    
    # # 1. 스케일링 누적 O
    # if scaler is None:
    #     x_train = x_train
    #     x_test = x_test
    #     test_csv = test_csv
    # else:
    #     scaler.fit(x_train)
    #     x_train = scaler.transform(x_train)
    #     x_test = scaler.transform(x_test)
    #     test_csv = scaler.transform(test_csv)

    # 2. 모델 재생성 (메모리 관리)
    tf.keras.backend.clear_session()

    # 2. 모델구성    
    model = Sequential()
    model.add(Dense(128, input_dim=200, activation='relu'))
    model.add(BatchNormalization()) # 레이어 출력값 정규화
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # 3. 컴파일, 훈련
    model.compile(loss='binary_crossentropy', 
                optimizer='adam',
                metrics=['AUC'],  # 보조지표 : Accuracy, AUC 등 (훈련에 영향X)
                ) 
    
    es = EarlyStopping( 
        monitor = 'val_auc',       
        mode = 'auto',               
        patience=50,             
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

    path = './_save/kaggle/12_santander/'
    # filepath 가변 (갱신때마다 저장)
    # filename = '{epoch:04d}-{val_auc:.4f}.hdf5'    # 04d : 정수 4자리, .4f : 소수점 4자리
    # filepath = "".join([path, 'k30_', date, '_', filename])     # 구분자를 공백("")으로 하겠다.
    # filepath 고정 (종료때만 저장)
    filepath = path + f'keras30_mcp3_{scaler_name}.hdf5'

    print(filepath)

    #exit()

    mcp = ModelCheckpoint(          # 모델+가중치 저장
        monitor = 'val_auc',
        mode = 'auto',
        save_best_only=True,
        filepath = filepath,
    )
    
    start_time = time.time()
    hist = model.fit(x_train, y_train, 
                    epochs=200, 
                    batch_size=128,
                    verbose=1, 
                    #validation_split=0.3,
                    validation_data=(x_val, y_val),
                    callbacks=[es, mcp],
                    #lass_weight=class_weight_dict,
                    )
    end_time = time.time()
    
    #################### 속도 측정 ####################
    if gpus:
        print('GPU 있다~')
    else:
        print('GPU 없다~')
    print("걸린 시간 :", round(end_time-start_time, 2), "초")
    # GPU 있다~
    # 걸린 시간 : 매~우 느림
    
    # GPU 없다~
    # 걸린 시간 : 5분이내
    #################### 속도 측정 ####################
    
    ## 그래프 그리기
    plt.figure(figsize=(18, 5))
    # 첫 번째 그래프
    plt.subplot(1, 2, 1)  # (행, 열, 위치)
    plt.plot(hist.history['loss'], c='red', label='loss')
    plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
    plt.title('snatander Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()

    # 두 번째 그래프
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['auc'], c='green', label='acc')
    plt.plot(hist.history['val_auc'], c='orange', label='val_acc')
    plt.title('snatander AUC')
    plt.xlabel('epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid()

    plt.tight_layout()  # 간격 자동 조정
    plt.show()
    
    # 4. 평가, 예측
    ## 4.1 평가
    results = model.evaluate(x_test, y_test)
    print(results)
    print("loss : ", results[0]) 
    print("auc : ", results[1]) 

    ## 예측(auc)
    y_pred = model.predict(x_test)
    #y_pred = np.round(y_pred)  # auc는 실수 값그대로 입력
    auc = roc_auc_score(y_test, y_pred)
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print("auc : ", auc) 
    
    ##### csv 파일 만들기 #####
    #y_pred = np.round(y_pred)
    y_submit = model.predict(test_csv)
    print(y_submit)

    submission_csv['target'] = y_submit
    from datetime import datetime
    path = './_data/kaggle/santander/'
    current_time = datetime.now().strftime('%y%m%d%H%M%S')
    submission_csv.to_csv(f'{path}submission_{current_time}_{scaler_name}.csv', index=False)  # 인덱스 생성옵션 끄면 첫번째 컬럼이 인덱스로 지정됨.(안끄면 인덱스 자동생성)

"""
    === 현재 적용 스케일러: None ===
    auc :  0.8551624031678131

    === 현재 적용 스케일러: MinMax ===
    auc :  0.857679927007482

    === 현재 적용 스케일러: Standard ===
    auc :  0.859737435275554

    === 현재 적용 스케일러: MaxAbs ===
    auc :  0.8567503565762216

    === 현재 적용 스케일러: Robust ===
    auc :  0.8583170017843048
    ================================================
    === 현재 적용 스케일러: None ===
    auc :  0.8555716936365809
    
    === 현재 적용 스케일러: MinMax ===
    auc :  0.856983437586268
    
    === 현재 적용 스케일러: Standard ===
    auc :  0.8562456732267886
    
    === 현재 적용 스케일러: MaxAbs ===
    auc :  0.8557852857950574
    
    === 현재 적용 스케일러: Robust ===
    auc :  0.8566520349128478
"""   