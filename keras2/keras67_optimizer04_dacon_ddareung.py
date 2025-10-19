### <<52>>

import tensorflow as tf 
print(tf.__version__)   # 2.7.4
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')

import sklearn as sk 
print(sk.__version__)
import tensorflow as tf 
print(tf.__version__)
import numpy as np

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
import pandas as pd

# 1. 데이터
path = './_data/dacon/따릉이/'          # 시스템 경로에서 시작.

train_csv =  pd.read_csv(path + 'train.csv', index_col=0)     # 0번컬럼을 인덱스컬럼으로 지정 -> 데이터프레임 컬럼에서 제거하고 인덱스로 지정해줌.
print(train_csv)        # [1459 rows x 11 columns] -> [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)  
# test_csv는 predict의 input으로 사용한다.
print(test_csv)         # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission_csv)   # [715 rows x 1 columns]

print(train_csv.shape)      # (1459, 10)
print(test_csv.shape)       # (715, 9)
print(submission_csv.shape) # (715, 1)
# train_csv : 학습데이터
# test_csv : 테스트데이터
# submission_csv : test_csv를 predict하여 예측한 값을 넣어서 제출 

print(train_csv.columns) 
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info())     # non-null수 확인(rows와 비교해서 결측치 수 확인), 데이터 타입 확인

print(train_csv.describe()) # 컬럼별 각종 정보확인할 수 있음 (평균,최댓값, 최솟값 등)

# 1. 데이터

######################################## 결측치 처리 1. 삭제 ########################################
# print(train_csv.isnull().sum())       # 컬럼별 결측치의 갯수 출력
print(train_csv.isna().sum())           # 컬럼별 결측치의 갯수 출력

# train_csv = train_csv.dropna()        # 결측치 제거
# print(train_csv.isna().sum())
# print(train_csv.info())
# print(train_csv)                      # [1328 rows x 10 columns]

######################################## 결측치 처리 2. 평균값 넣기 ########################################
train_csv = train_csv.fillna(train_csv.mean())
print(train_csv.isna().sum())
print(train_csv.info())

########################################  test_csv 결측치 확인 및 처리 ########################################
# test_csv는 결측치 있을 경우 절대 삭제하면 안된다. 답안지에 해당하는(submission_csv)에 채워넣으려면 갯수가 맞아야한다.
print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean())
print('test_csv 정보:', test_csv)
print('ㅡㅡㅡㅡㅡㅡ')

x = train_csv.drop(['count'], axis=1)   # axis = 1 : 컬럼 // axis = 0 : 행
print(x)    # [1459 rows x 9 columns] : count 컬럼을 제거

y = train_csv['count']      # count 컬럼만 추출
print(y.shape)  # (1469,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=1111,
)

# 스케일러
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 파라미터튜닝
optimizers = [Adam, Adagrad, SGD, RMSprop]
learning_rates = [0.1, 0.01, 0.05, 0.001, 0.0001]
# 파라미터튜닝에서 가장 성능차이가 커지는 파라미터는 learning_rate

# 출력용 파라미터
best_score = -float('inf')
best_optim = None
best_lr = None

# 2. 모델구성
for optim in optimizers:
    for lr in learning_rates:
        model = Sequential()
        model.add(Dense(128, input_dim=9))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='linear')) 

        # 3. 컴파일, 훈련
        model.compile(loss='mse', optimizer=optim(learning_rate=lr))

        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        es = EarlyStopping( 
            monitor = 'val_loss',       
            mode = 'min',               
            patience=30,             
            restore_best_weights=True,  
        )
        start_time = time.time()
        hist = model.fit(x_train, y_train, 
                        epochs=100, 
                        batch_size=32, 
                        verbose=0, 
                        validation_split=0.2,
                        callbacks=[es],
                        ) 
        end_time = time.time()

        #################### 속도 측정 ####################
        if gpus:
            print('GPU 있다~')
        else:
            print('GPU 없다~')
        print("걸린 시간 :", round(end_time-start_time, 2), "초")
        #################### 속도 측정 ####################
            
        # 4. 평가, 예측
        loss = model.evaluate(x_test, y_test)
        results = model.predict(x_test)

        from sklearn.metrics import r2_score, mean_absolute_error
        try:
            r2_score = r2_score(y_test, results)
        except:
            r2_score = "Nan"
            
        # 최고값 갱신
        if r2_score > best_score:
            best_score = r2_score
            best_optim = optim.__name__
            best_lr = lr
        
        print(f'{optim.__name__},  {lr} 일때의 r2 스코어 :', r2_score)
        print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
        
print("=======================================")
print(f'최고 r2 스코어 : {best_score:.4f}')
print(f'최적 optimizer : {best_optim}')
print(f'최적 learning_rate : {best_lr}')
print("=======================================")

# Adam,  0.1 일때의 r2 스코어 : 0.6887227236889607
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.48 초
# 14/14 [==============================] - 0s 1ms/step - loss: 2285.0984
# Adam,  0.01 일때의 r2 스코어 : 0.6787590118683529
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.48 초
# 14/14 [==============================] - 0s 1ms/step - loss: 2151.2285
# Adam,  0.05 일때의 r2 스코어 : 0.6975785165571511
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.44 초
# 14/14 [==============================] - 0s 0s/step - loss: 2248.2454
# Adam,  0.001 일때의 r2 스코어 : 0.6839398492047937
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.42 초
# 14/14 [==============================] - 0s 841us/step - loss: 2751.9043
# Adam,  0.0001 일때의 r2 스코어 : 0.6131350930391024
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.33 초
# 14/14 [==============================] - 0s 155us/step - loss: 2067.2742
# Adagrad,  0.1 일때의 r2 스코어 : 0.7093809178134899
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.25 초
# 14/14 [==============================] - 0s 131us/step - loss: 2504.3689
# Adagrad,  0.01 일때의 r2 스코어 : 0.6479337931078502
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.29 초
# 14/14 [==============================] - 0s 549us/step - loss: 2190.2844
# Adagrad,  0.05 일때의 r2 스코어 : 0.692088028012029
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.22 초
# 14/14 [==============================] - 0s 0s/step - loss: 4375.0605
# Adagrad,  0.001 일때의 r2 스코어 : 0.38495042698882864
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.22 초
# 14/14 [==============================] - 0s 410us/step - loss: 19047.9375
# Adagrad,  0.0001 일때의 r2 스코어 : -1.6777746868062748
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.16 초
# 14/14 [==============================] - 0s 557us/step - loss: nan
# SGD,  0.1 일때의 r2 스코어 : Nan
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.18 초
# 14/14 [==============================] - 0s 1ms/step - loss: nan
# SGD,  0.01 일때의 r2 스코어 : Nan
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.12 초
# 14/14 [==============================] - 0s 465us/step - loss: nan
# SGD,  0.05 일때의 r2 스코어 : Nan
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.11 초
# 14/14 [==============================] - 0s 759us/step - loss: nan
# SGD,  0.001 일때의 r2 스코어 : Nan
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.11 초
# 14/14 [==============================] - 0s 0s/step - loss: 2124.2620
# SGD,  0.0001 일때의 r2 스코어 : 0.7013695229301972
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.5 초
# 14/14 [==============================] - 0s 0s/step - loss: 7113.4639
# RMSprop,  0.1 일때의 r2 스코어 : -1.650942937003741e-05
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.43 초
# 14/14 [==============================] - 0s 232us/step - loss: 2083.2991
# RMSprop,  0.01 일때의 r2 스코어 : 0.7071281345053861
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.41 초
# 14/14 [==============================] - 0s 1ms/step - loss: 2348.1958
# RMSprop,  0.05 일때의 r2 스코어 : 0.6698887383384498
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.37 초
# 14/14 [==============================] - 0s 808us/step - loss: 2294.2971
# RMSprop,  0.001 일때의 r2 스코어 : 0.6774658174104278
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 3.38 초
# 14/14 [==============================] - 0s 461us/step - loss: 2778.6572
# RMSprop,  0.0001 일때의 r2 스코어 : 0.6093740949417167
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
