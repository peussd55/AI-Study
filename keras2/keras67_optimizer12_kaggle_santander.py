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
from tensorflow.keras.layers import Dropout, BatchNormalization
import time
import pandas as pd

# 1. 데이터
path = './_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=1111,
    stratify=y,
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
        model.compile(loss='binary_crossentropy', optimizer=optim(learning_rate=lr))

        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        es = EarlyStopping( 
            monitor = 'val_loss',       
            mode = 'min',               
            patience=30,             
            restore_best_weights=True,  
        )
        start_time = time.time()
        hist = model.fit(x_train, y_train, 
                        epochs=10, 
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
        
        from sklearn.metrics import r2_score, accuracy_score
        try:
            acc = accuracy_score(y_test, np.round(results))
        except:
            acc = "Nan"
            
        # 최고값 갱신
        if acc > best_score:
            best_score = acc
            best_optim = optim.__name__
            best_lr = lr
        
        print(f'{optim.__name__},  {lr} 일때의 acc 스코어 :', acc)
        print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
        
print("=======================================")
print(f'최고 r2 스코어 : {best_score:.4f}')
print(f'최적 optimizer : {best_optim}')
print(f'최적 learning_rate : {best_lr}')
print("=======================================")

# GPU 없다~
# 걸린 시간 : 32.18 초
# 1875/1875 [==============================] - 1s 544us/step - loss: 0.3262
# Adam,  0.1 일때의 acc 스코어 : 0.8995166666666666
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 31.95 초
# 1875/1875 [==============================] - 1s 547us/step - loss: 0.2419
# Adam,  0.01 일때의 acc 스코어 : 0.914
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 32.02 초
# 1875/1875 [==============================] - 1s 540us/step - loss: 0.2674
# Adam,  0.05 일때의 acc 스코어 : 0.8995166666666666
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 32.75 초
# 1875/1875 [==============================] - 1s 553us/step - loss: 0.2390
# Adam,  0.001 일때의 acc 스코어 : 0.9117666666666666
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 32.53 초
# 1875/1875 [==============================] - 1s 531us/step - loss: 0.2354
# Adam,  0.0001 일때의 acc 스코어 : 0.9145333333333333
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 30.35 초
# 1875/1875 [==============================] - 1s 551us/step - loss: 0.2369
# Adagrad,  0.1 일때의 acc 스코어 : 0.9132833333333333
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 30.87 초
# 1875/1875 [==============================] - 1s 534us/step - loss: 0.2361
# Adagrad,  0.01 일때의 acc 스코어 : 0.91365
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 30.99 초
# 1875/1875 [==============================] - 1s 525us/step - loss: 0.2354
# Adagrad,  0.05 일때의 acc 스코어 : 0.9143666666666667
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 30.54 초
# 1875/1875 [==============================] - 1s 551us/step - loss: 0.2979
# Adagrad,  0.001 일때의 acc 스코어 : 0.89955
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 33.43 초
# 1875/1875 [==============================] - 1s 594us/step - loss: 0.6180
# Adagrad,  0.0001 일때의 acc 스코어 : 0.6606166666666666
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 31.63 초
# 1875/1875 [==============================] - 1s 575us/step - loss: 0.2362
# SGD,  0.1 일때의 acc 스코어 : 0.9147
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 29.86 초
# 1875/1875 [==============================] - 1s 517us/step - loss: 0.2349
# SGD,  0.01 일때의 acc 스코어 : 0.9145666666666666
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 30.11 초
# 1875/1875 [==============================] - 1s 542us/step - loss: 0.2335
# SGD,  0.05 일때의 acc 스코어 : 0.9147666666666666
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 30.01 초
# 1875/1875 [==============================] - 1s 556us/step - loss: 0.2979
# SGD,  0.001 일때의 acc 스코어 : 0.8995166666666666
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 31.05 초
# 1875/1875 [==============================] - 1s 569us/step - loss: 0.3516
# SGD,  0.0001 일때의 acc 스코어 : 0.8995166666666666
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 34.32 초
# 1875/1875 [==============================] - 1s 545us/step - loss: 0.3226
# RMSprop,  0.1 일때의 acc 스코어 : 0.8995166666666666
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 33.66 초
# 1875/1875 [==============================] - 1s 549us/step - loss: 0.2483
# RMSprop,  0.01 일때의 acc 스코어 : 0.9095833333333333
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 34.42 초
# 1875/1875 [==============================] - 1s 546us/step - loss: 0.3021
# RMSprop,  0.05 일때의 acc 스코어 : 0.9095666666666666
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 33.57 초
# 1875/1875 [==============================] - 1s 560us/step - loss: 0.2503
# RMSprop,  0.001 일때의 acc 스코어 : 0.9103833333333333
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 없다~
# 걸린 시간 : 38.75 초
# 1875/1875 [==============================] - 1s 653us/step - loss: 0.2463
# RMSprop,  0.0001 일때의 acc 스코어 : 0.9111
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ