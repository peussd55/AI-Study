### <<39>>

### SMOTE의 문제점 : 범주형변수 x컬럼을 증폭시키면 보간법에 의해 알수없는 값이 라벨된다.(ex : 0 또는 1로 라벨된 x변수를 SMOTE하면 보간법에 의해 0과 1사이의 값이 부여된다.)

# https://www.kaggle.com/c/otto-group-product-classification-challenge/overview
# 다중분류

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
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf 
print(tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

# 1. 데이터
path = './_data/kaggle/otto/'

# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

print(train_csv.shape, test_csv.shape, submission_csv.shape)  # (61878, 94) (144368, 93) (144368, 10)
print(train_csv.head())
print(test_csv.head())

print(train_csv.columns)
# Index(['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7',
#         ...
#        'feat_92', 'feat_93', 'target'],
print(test_csv.columns)
# Index(['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7',
#         ...
#        'feat_92', 'feat_93'],

# 결측치 확인
print(train_csv.info())
print(train_csv.isna().sum())       # 결측치 없음
print(test_csv.info())
print(test_csv.isna().sum())       # 결측치 없음

print(train_csv['target'].value_counts())
# 다중분류(불균형)
# Class_2    16122
# Class_6    14135
# Class_8     8464
# Class_3     8004
# Class_9     4955
# Class_7     2839
# Class_5     2739
# Class_4     2691
# Class_1     1929

# target컬럼 레이블 인코딩(원핫 인코딩 사전작업)###
# 정수형을 직접 원핫인코딩할경우 keras, pandas, sklearn 방식 모두 가능하지만 문자형태로 되어있을 경우에는 pandas방식만 문자열에서 직접 원핫인코딩이 가능하다.
le = LabelEncoder() # 인스턴스화
train_csv['target'] = le.fit_transform(train_csv['target'])   # fit 함수 + transform 함친 합친 함수 : 변환해서 적용

# train data x로 y 분리
x = train_csv.drop(['target'], axis=1)
print(x)
print('x type:',type(x))
y = train_csv['target']
print('y type:',type(y))

# # onehotencoding
# y = pd.get_dummies(y)
# print("Pandas get_dummies:")
# print(y.head())
# print(y.head(-5))    
# print(y.shape)      # (61878, 9)
# print(type(y))      # <class 'pandas.core.frame.DataFrame'>

from imblearn.over_sampling import SMOTENC, SMOTE
# SMOTENC 적용 : 범주형 컬럼의 데이터 증폭시킬때 0.xx 같은 값이 아니라 0 또는 1 중에 부여한다.(투표방식으로)
# SMOTENC를 쓴다고 꼭 성능이 올라가는 것은 아니다. 어차피 회귀모델로 학습을 한다면 범주형데이터라도 실수로 계산하고 값을 도출한다.
# SMOTENC는 수치형과 범주형이 모두 있는 경우에만 사용 가능, 한쪽이 모두 수치형이라면 SMOTE사용해야함.
# 범주형데이터만 있는 경우 : RandomOverSampler 사용

smote = SMOTE(random_state=333, 
               k_neighbors=5, # 디폴트

            )
x_res, y_res = smote.fit_resample(x, y)

print("Before:", y.value_counts())
print("After:", pd.Series(y_res).value_counts())

# 데이터분할
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    # x_res,y_res,
    train_size=0.7,
    random_state=8282,
    shuffle=True,
    stratify=y,
    # stratify=y_res,
)

# 훈련데이터에서 검증데이터 분할(검증데이터는 스케일링 적용되면 안되기때문에 여기서 분할)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, 
    test_size=0.2,
    random_state=5234,
    stratify=y_train,
    )

""" 
(2) 클래스가중치 적용방식 : 클래스 가중치를 적용하려면 원핫인코딩되기전 정수형태 데이터(y_original)를 가공해야한다
==> 성능처참해서 제외.
# 1. 데이터 준비
le = LabelEncoder()
train_csv['target'] = le.fit_transform(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
y_original = train_csv['target']  # 정수 레이블

# 데이터 분할
x_train, x_test, y_train_original, y_test_original = train_test_split(
    x, y_original,
    train_size=0.7,
    random_state=8282,
    shuffle=True,
    stratify=y_original
)

# 원-핫 인코딩
y_train = pd.get_dummies(y_train_original)
y_test = pd.get_dummies(y_test_original)

# 클래스 가중치 계산
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train_original),
    y=y_train_original
)
class_weight_dict = dict(enumerate(class_weights))
"""

scalers = [
    # ('None', None),
    # ('MinMax', MinMaxScaler()),
    ('Standard', StandardScaler()),
    # ('MaxAbs', MaxAbsScaler()),
    # ('Robust', RobustScaler()),
]

# 원본 데이터 백업
original_x_train = x_train.copy()
original_x_test = x_test.copy()
original_x_val = x_val.copy()
original_test_csv = test_csv.copy()

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
    model.add(Dense(512, input_dim=93, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(9, activation='softmax'))

    # 3. 컴파일, 훈련
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['acc'],
    )
    
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=30,
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

    path = './_save/kaggle/13_otto/'
    # filepath 가변 (갱신때마다 저장)
    # filename = '{epoch:04d}-{val_auc:.4f}.hdf5'    # 04d : 정수 4자리, .4f : 소수점 4자리
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

    start_time = time.time()
    hist = model.fit(x_train, y_train, 
                    epochs=100, 
                    batch_size=64,
                    verbose=3, 
                    # validation_split=0.2,
                    validation_data=(x_val, y_val),
                    callbacks=[es, mcp],
                    #class_weight=class_weight_dict,
                    )
    end_time = time.time()

    #################### 속도 측정 ####################
    if gpus:
        print('GPU 있다~')
    else:
        print('GPU 없다~')
    print("걸린 시간 :", round(end_time-start_time, 2), "초")
    # GPU 있다~
    # 걸린 시간 : 100초 대
    
    # GPU 없다~
    # 걸린 시간 : 100초 대
    #################### 속도 측정 ####################

    ## 그래프 그리기
    # plt.figure(figsize=(18, 5))
    plt.figure(figsize=(9,6))       # 9 x 6 사이즈
    # # 첫 번째 그래프
    # plt.subplot(1, 2, 1)  # 서브플롯추가(행, 열, 위치)
    plt.plot(hist.history['loss'], c='red', label='loss')
    plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
    plt.title('otto Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()  # 간격 자동 조정
    plt.show()

    # 4. 평가, 예측
    results = model.evaluate(x_test, y_test)
    print(results)
    print("loss : ", results[0]) 
    
    # 그냥
    # loss :  0.5061825513839722
    
    # [SMOTE]
    # loss :  0.3250315487384796

    ## 예측 : 평가지표가 categorical_crossentropy 손실계산이기때문에 별도로 예측결과를 후처리하지않는다.
    #y_pred = model.predict(x_test)

    ##### csv 파일 만들기 #####
    y_submit = model.predict(test_csv)
    print(y_submit.shape)  # (144368, 9)

    # 클래스 컬럼명 생성 (예: Class_1 ~ Class_9)
    class_columns = [f'Class_{i+1}' for i in range(9)]

    # 예측값을 각 클래스 컬럼에 할당
    submission_csv[class_columns] = y_submit

    # 파일 저장
    from datetime import datetime
    path = './_data/kaggle/otto/'
    current_time = datetime.now().strftime('%y%m%d%H%M%S')
    submission_csv.to_csv(f'{path}submission_{current_time}_{scaler_name}_{results[0]}.csv', index=False)

