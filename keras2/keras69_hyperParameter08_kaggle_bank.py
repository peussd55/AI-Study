### <<52>>

# kaggle_bike

# kaggle_bank

# kaggle_otto

# fashion mnist

# jena

import numpy as np 
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
import pandas as pd

# 1. 데이터
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 인코딩
le_geo = LabelEncoder()
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

# 불필요 컬럼 제거
train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=1111
)
print(x_train.shape, y_train.shape)

# 스케일러
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001,
                ):
    inputs = Input(shape=(10,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='sigmoid', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'], loss='binary_crossentropy')
    
    return model

def create_hyperparameter():
    batchs =  [32, 16, 8, 1, 64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16, 8]
    epochs = [10, 20, 50, 100]
    # epochs = [1]
    
    return {
        'batch_size' : batchs,
        'optimizer' : optimizers,
        'drop' : dropouts,
        'activation' : activations,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5,
        'epochs' : epochs,
    }

hyperparameter = create_hyperparameter()
print(hyperparameter)
# {'batch_size': [32, 16, 8, 1, 64], 'optimizer': ['adam', 'rmsprop', 'adadelta'], 'drop': [0.2, 0.3, 0.4, 0.5], 'activation': ['relu', 'elu', 'selu', 'linear'], 
# 'node1': [128, 64, 32, 16], 'node2': [128, 64, 32, 16], 'node3': [128, 64, 32, 16], 'node4': [128, 64, 32, 16], 'node5': [128, 64, 32, 16, 8], 'epochs': [10, 20, 50, 100]} 

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# sklearn RandomizedSearchCV가 인식할 수있는 모델객체로 wrapping해야함.
keras_model = KerasRegressor(build_fn=build_model, verbose=1)
model = RandomizedSearchCV(keras_model, hyperparameter, cv=5,
                           n_iter=1, 
                           verbose=10,
                            #    refit=False, : 최적의 파라미터만 찾아줌. 최종 모델(best_estimator_)을 저장하지 않으므로 predict사용불가.
                            #    refit=True, : 디폴트. 최적의 파라미터로 최종 모델(best_estimator_)을 저장. predict사용가능.
                           )

# 3. 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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
start = time.time()
# fit_params = {
#     'callbacks': [es, rlr],
#     'validation_split': 0.1
# }
# model.fit(x_train, y_train, **fit_params)
model.fit(x_train, y_train, callbacks=[es, rlr], validation_split=0.1)
end = time.time()

# print('최적의 매개변수 :', model.best_estimator_.get_params())
print('최적의 파라미터 :', model.best_params_) 

# 4. 평가, 예측
print('best_score :', model.best_score_)                # (x_train)에서의 교차검증 평균 성능
print('model.score :', model.score(x_test, y_test))     # 실제 테스트 데이터에서의 모델 성능

y_pred = model.predict(x_test)
print('ACC :', accuracy_score(y_test, np.round(y_pred)))

y_pred_best = model.best_estimator_.predict(x_test)     # refit=True 이면 predict나 best_estimator_.predict나 차이없다 : 이미 refit=True 옵션으로 최적의 가중치를 찾아놨기때문.
print('ACC :', accuracy_score(y_test, np.round(y_pred_best)))

print('걸린시간 :', round(end - start), '초')

# 총 epoch횟수 : cv * n_iter * epochs + 1(최종모델)

"""
Epoch 1/10
95058/95058 [==============================] - 57s 599us/step - loss: 0.5192 - acc: 0.7653 - val_loss: 0.4369 - val_acc: 0.8053 - lr: 0.0010
Epoch 2/10
95058/95058 [==============================] - 56s 584us/step - loss: 0.4433 - acc: 0.8046 - val_loss: 0.4155 - val_acc: 0.8147 - lr: 0.0010
Epoch 3/10
95058/95058 [==============================] - 55s 579us/step - loss: 0.4360 - acc: 0.8080 - val_loss: 0.4126 - val_acc: 0.8171 - lr: 0.0010
Epoch 4/10
95058/95058 [==============================] - 58s 606us/step - loss: 0.4332 - acc: 0.8125 - val_loss: 0.4122 - val_acc: 0.8197 - lr: 0.0010
Epoch 5/10
95058/95058 [==============================] - 53s 558us/step - loss: 0.4306 - acc: 0.8141 - val_loss: 0.4123 - val_acc: 0.8206 - lr: 0.0010
Epoch 6/10
95058/95058 [==============================] - 53s 560us/step - loss: 0.4301 - acc: 0.8150 - val_loss: 0.4124 - val_acc: 0.8215 - lr: 0.0010
Epoch 7/10
95058/95058 [==============================] - 53s 560us/step - loss: 0.4284 - acc: 0.8176 - val_loss: 0.4125 - val_acc: 0.8218 - lr: 0.0010
Epoch 8/10
95058/95058 [==============================] - 55s 578us/step - loss: 0.4276 - acc: 0.8169 - val_loss: 0.4125 - val_acc: 0.8220 - lr: 0.0010
Epoch 9/10
95058/95058 [==============================] - 56s 585us/step - loss: 0.4266 - acc: 0.8176 - val_loss: 0.4126 - val_acc: 0.8221 - lr: 0.0010
Epoch 10/10
95058/95058 [==============================] - 55s 578us/step - loss: 0.4247 - acc: 0.8192 - val_loss: 0.4127 - val_acc: 0.8224 - lr: 0.0010
26406/26406 [==============================] - 11s 432us/step - loss: 0.4032 - acc: 0.8270
[CV 1/5; 1/1] END activation=linear, batch_size=1, drop=0.3, epochs=10, node1=128, node2=64, node3=64, node4=16, node5=128, optimizer=adadelta;, score=-0.403 total time= 9.4min
[CV 2/5; 1/1] START activation=linear, batch_size=1, drop=0.3, epochs=10, node1=128, node2=64, node3=64, node4=16, node5=128, optimizer=adadelta
...
[CV 2/5; 1/1] END activation=linear, batch_size=1, drop=0.3, epochs=10, node1=128, node2=64, node3=64, node4=16, node5=128, optimizer=adadelta;, score=-0.410 total time= 9.4min
[CV 3/5; 1/1] START activation=linear, batch_size=1, drop=0.3, epochs=10, node1=128, node2=64, node3=64, node4=16, node5=128, optimizer=adadelta
...
[CV 3/5; 1/1] END activation=linear, batch_size=1, drop=0.3, epochs=10, node1=128, node2=64, node3=64, node4=16, node5=128, optimizer=adadelta;, score=-0.400 total time= 9.6min
[CV 4/5; 1/1] START activation=linear, batch_size=1, drop=0.3, epochs=10, node1=128, node2=64, node3=64, node4=16, node5=128, optimizer=adadelta
...
[CV 3/5; 1/1] END activation=linear, batch_size=1, drop=0.3, epochs=10, node1=128, node2=64, node3=64, node4=16, node5=128, optimizer=adadelta;, score=-0.400 total time= 9.6min
[CV 4/5; 1/1] START activation=linear, batch_size=1, drop=0.3, epochs=10, node1=128, node2=64, node3=64, node4=16, node5=128, optimizer=adadelta
...
"""