### <<52>>

import numpy as np 
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import time

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=1111
)
print(x_train.shape, y_train.shape)

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
    outputs = Dense(1, activation='linear', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    
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
                           n_iter=10, 
                           verbose=1,
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
fit_params = {
    'callbacks': [es, rlr],
    'validation_split': 0.1
}
model.fit(x_train, y_train, **fit_params)
end = time.time()

print('최적의 매개변수 :', model.best_estimator_.get_params())
print('최적의 파라미터 :', model.best_params_) 

# 4. 평가, 예측
print('best_score :', model.best_score_)                # (x_train)에서의 교차검증 평균 성능
print('model.score :', model.score(x_test, y_test))     # 실제 테스트 데이터에서의 모델 성능

y_pred = model.predict(x_test)
print('r2_score :', r2_score(y_test, y_pred))

y_pred_best = model.best_estimator_.predict(x_test)     # refit=True 이면 predict나 best_estimator_.predict나 차이없다 : 이미 refit=True 옵션으로 최적의 가중치를 찾아놨기때문.
print('r2_score :', r2_score(y_test, y_pred_best))

print('걸린시간 :', round(end - start), '초')

"""
최적의 매개변수 : {'verbose': 1, 'optimizer': 'adam', 'node5': 32, 'node4': 16, 'node3': 64, 'node2': 16, 'node1': 32, 'epochs': 100, 'drop': 0.2, 'batch_size': 1, 'activation': 'selu', 'build_fn': <function build_model at 0x000002A767A2AD30>}
최적의 파라미터 : {'optimizer': 'adam', 'node5': 32, 'node4': 16, 'node3': 64, 'node2': 16, 'node1': 32, 'epochs': 100, 'drop': 0.2, 'batch_size': 1, 'activation': 'selu'}
best_score : -3268.77451171875
89/89 [==============================] - 0s 373us/step - loss: 2522.5239 - mae: 40.1653
model.score : -2522.52392578125
89/89 [==============================] - 0s 415us/step
r2_score : 0.5861126162510437
89/89 [==============================] - 0s 397us/step
r2_score : 0.5861126162510437
걸린시간 : 253 초
"""