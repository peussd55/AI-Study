### <<52>>

# kaggle_bike

# kaggle_bank

# kaggle_otto

# fashion mnist

# jena

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 1. 데이터
path = './_data/kaggle/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# 데이터 분리
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

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

# 2. 모델구성

model = Sequential()
model.add(Dense(128, input_dim=8))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear')) 

# 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer=optim(learning_rate=lr)) 
optimizer = Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer=optimizer)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping( 
    monitor = 'val_loss',       
    mode = 'min',               
    patience=30,             
    verbose=1,
    restore_best_weights=True,  
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=10,    # 10번 동안 val_loss가 갱신되지 않으면
    verbose=1,
    factor=0.5,     # lr을 해당 비율만큼 곱해 줄인다
)
start_time = time.time()
hist = model.fit(x_train, y_train, 
                epochs=100, 
                batch_size=32, 
                verbose=1, 
                validation_split=0.2,
                callbacks=[es, rlr],
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
    r2 = r2_score(y_test, results)
except:
    r2 = "Nan"

current_lr = float(model.optimizer.learning_rate.numpy())
print(f'{model.optimizer.__class__.__name__}, {current_lr:.4f} 일때의 r2 스코어 :', r2)

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
# 반복문내에서 같은 print문을 2번이상쓰면 버퍼가 꼬여서 제대로 출력안되는 경우 있음

# Adam, 0.0001 일때의 r2 스코어 : 0.8006169627484749