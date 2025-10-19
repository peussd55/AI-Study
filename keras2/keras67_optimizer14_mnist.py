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
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dropout, BatchNormalization
import time
import pandas as pd


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

#  x reshape -> (60000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

# 원핫인코딩
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# 파라미터튜닝
optimizers = [Adam, Adagrad, SGD, RMSprop]
learning_rates = [0.1, 0.01, 0.05, 0.001, 0.0001]
# 파라미터튜닝에서 가장 성능차이가 커지는 파라미터는 learning_rate

# 평가, 예측때 argmax안된 y_test불러오기 위한 변수설정(값만 복사해와야함) -> nparray로 변환
y_test_origin = y_test.copy().values

# 출력용 파라미터
best_score = -float('inf')
best_optim = None
best_lr = None


# 2. 모델구성
for optim in optimizers:
    for lr in learning_rates:
        model = Sequential()
        model.add(Conv2D(64, (3,3), strides=1, input_shape=(28, 28, 1)))    # (28, 28, 1) 에서의 채널(1 또는 3)은 여기서만 쓰이고 다음 레이어부터는 filters를 입력받는다. 
        model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
        model.add(Dropout(0.2))              
        model.add(Conv2D(32, (3,3), activation='relu'))         # activation의 역할 : 레이어의 출력을 한정시킴
        model.add(Flatten())                                    # Flatten 하는 이유 : 다중분류를 위해선 softmax를 써야하고 softmax를 쓰기위해선 faltten으로 차원변환을 해야하기때문.
        model.add(Dense(units=16, activation='relu'))           # 원하는 평가를 하기위해서 그에 맞는 데이터타입변환 하기위해 Flatten을 사용한다.
        model.add(Dropout(0.2))
        model.add(Dense(units=16, input_shape=(16,)))
        model.add(Dense(units=10, activation='softmax'))

        # 3. 컴파일, 훈련
        model.compile(loss='categorical_crossentropy', optimizer=optim(learning_rate=lr))   # y원핫인코딩되었으므로 sparse_categorical_crossentropy 사용 불가

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
                        batch_size=128, 
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
        # 평가
        loss = model.evaluate(x_test, y_test_origin)
        
        # 예측
        y_pred = model.predict(x_test)
        
        # argmax(axis=1) : 열 중에서 가장큰 값을 가진 열의 인덱스를 반환
        y_pred_label = np.argmax(y_pred, axis=1)
        y_test_label = np.argmax(y_test_origin, axis=1)
        
        print(y_pred_label.shape, y_test_label.shape)   # (10000,) (10000,)
    
        from sklearn.metrics import r2_score, accuracy_score
        try:
            acc = accuracy_score(y_test_label, y_pred_label)
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

# GPU 있다~
# 걸린 시간 : 60.17 초
# 313/313 [==============================] - 1s 2ms/step - loss: 2.3120
# (10000,) (10000,)
# Adam,  0.1 일때의 acc 스코어 : 0.1135
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 66.4 초
# 313/313 [==============================] - 1s 3ms/step - loss: 0.1668
# (10000,) (10000,)
# Adam,  0.01 일때의 acc 스코어 : 0.958
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 67.46 초
# 313/313 [==============================] - 1s 2ms/step - loss: 2.3063
# (10000,) (10000,)
# Adam,  0.05 일때의 acc 스코어 : 0.1028
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 49.11 초
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0417
# (10000,) (10000,)
# Adam,  0.001 일때의 acc 스코어 : 0.9876
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 44.4 초
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0602
# (10000,) (10000,)
# Adam,  0.0001 일때의 acc 스코어 : 0.9825
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 67.2 초
# 313/313 [==============================] - 1s 3ms/step - loss: 0.0859
# (10000,) (10000,)
# Adagrad,  0.1 일때의 acc 스코어 : 0.9791
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 67.57 초
# 313/313 [==============================] - 1s 3ms/step - loss: 0.0619
# (10000,) (10000,)
# Adagrad,  0.01 일때의 acc 스코어 : 0.9806
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 52.61 초
# 313/313 [==============================] - 1s 3ms/step - loss: 0.0442
# (10000,) (10000,)
# Adagrad,  0.05 일때의 acc 스코어 : 0.9876
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 47.37 초
# 313/313 [==============================] - 1s 2ms/step - loss: 0.2858
# (10000,) (10000,)
# Adagrad,  0.001 일때의 acc 스코어 : 0.9201
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 47.64 초
# 313/313 [==============================] - 1s 3ms/step - loss: 1.4452
# (10000,) (10000,)
# Adagrad,  0.0001 일때의 acc 스코어 : 0.5376
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 49.11 초
# 313/313 [==============================] - 1s 3ms/step - loss: 0.0408
# (10000,) (10000,)
# SGD,  0.1 일때의 acc 스코어 : 0.9872
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 46.73 초
# 313/313 [==============================] - 1s 2ms/step - loss: 0.1318
# (10000,) (10000,)
# SGD,  0.01 일때의 acc 스코어 : 0.9608
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 44.05 초
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0490
# (10000,) (10000,)
# SGD,  0.05 일때의 acc 스코어 : 0.9853
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 51.05 초
# 313/313 [==============================] - 1s 2ms/step - loss: 0.3413
# (10000,) (10000,)
# SGD,  0.001 일때의 acc 스코어 : 0.9075
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 48.86 초
# 313/313 [==============================] - 1s 2ms/step - loss: 2.1753
# (10000,) (10000,)
# SGD,  0.0001 일때의 acc 스코어 : 0.2335
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 57.08 초
# 313/313 [==============================] - 1s 2ms/step - loss: 2.3122
# (10000,) (10000,)
# RMSprop,  0.1 일때의 acc 스코어 : 0.1032
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 70.66 초
# 313/313 [==============================] - 1s 2ms/step - loss: 2.3025
# (10000,) (10000,)
# RMSprop,  0.01 일때의 acc 스코어 : 0.1135
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 66.58 초
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3106
# (10000,) (10000,)
# RMSprop,  0.05 일때의 acc 스코어 : 0.0974
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 59.06 초
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0388
# (10000,) (10000,)
# RMSprop,  0.001 일때의 acc 스코어 : 0.9897
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# GPU 있다~
# 걸린 시간 : 60.08 초
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0729
# (10000,) (10000,)
# RMSprop,  0.0001 일때의 acc 스코어 : 0.9787
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
