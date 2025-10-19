# ### <<35>>

# from tensorflow.keras.datasets import mnist
# import  numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier

# # (x_train, y_train), (x_test, y_test) = mnist.load_data()
# (x_train, _), (x_test, _) = mnist.load_data()   # y_train, y_test는 받지 않겠다
# print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

# x = np.concatenate([x_train, x_test], axis=0)
# print(x.shape)  # (70000, 28, 28)

# x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# print(x.shape)  # (70000, 784)

# pca = PCA(n_components=x.shape[1])
# x = pca.fit_transform(x)

# evr = pca.explained_variance_ratio_
# evr_cumsum = np.cumsum(evr) # 누적합
# print(evr_cumsum)
# print(len(evr_cumsum))  # 784

# # 조건별 필요한 주성분 개수 찾기 함수
# def find_n_components(evr_cumsum, threshold, tolerance=1e-10):
#     """특정 임계값을 처음 넘는 주성분 개수를 반환"""
#     indices = np.where(evr_cumsum >= (threshold - tolerance))[0]
#     if len(indices) > 0:
#         return indices[0] + 1  # 인덱스는 0부터 시작하므로 +1
#     else:
#         return len(evr_cumsum)  # 모든 주성분이 필요

# # 1. 1.0일 때 몇개?
# tolerance = 1e-10
# n_comp_100 = find_n_components(evr_cumsum, 1.0 ,tolerance)
# print(784-n_comp_100, '개')   # 71 개
# # 2. 0.999 이상 몇개?
# n_comp_999 = find_n_components(evr_cumsum, 0.999)
# print(784-n_comp_999, '개')   # 298 개
# # 3. 0.99 이상 몇개?
# evr_cumsum_99 = find_n_components(evr_cumsum, 0.99)
# print(784-evr_cumsum_99, '개')  # 453 개
# # 4. 0.95 이상 몇개?
# evr_cumsum_95 = find_n_components(evr_cumsum, 0.95)
# print(784-evr_cumsum_95, '개')  # 630 개

#
num = [152, 331, 486, 713, 784]
# train_test_split => scaling => pca

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)
# print(type(x_train.shape))              # <class 'tuple'>
# print(x_train.shape[0]) # 60000
# print(x_train.shape[1]) # 28
# print(x_train.shape[2]) # 28
# print(x_train.shape[3]) # IndexError: tuple index out of range

# 스케일링
x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

# # y 원핫인코딩 (loss = sparse_categorical_crossentropy로 대체)
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# # y_train, y_test 매트릭스로 변환 : ohe가 입력으로 매트릭스 받기 때문
# y_train = y_train.reshape(60000, 1)
# y_test = y_test.reshape(-1, 1)  # -1 : 맨 마지막 인덱스 (=10000) : (10000, 1)
# print(y_train.shape, y_test.shape)  # (60000, 1) (10000, 1)

# y_train = ohe.fit_transform(y_train)
# y_test = ohe.fit_transform(y_test)
# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

num = [152, 331, 486, 713, 784]

for i in num: 
    x_train1 = x_train.copy()
    x_test1 = x_test.copy()

    pca = PCA(n_components=i)   # n_componets : 몇 개의 컬럼으로 압축할 것인지
    x_train1 = pca.fit_transform(x_train1)
    x_test1 = pca.transform(x_test1)

    # 2. 모델구성 // 성능 0.98이상 // 시간체크(cnn 방식이랑 시간 비교)
    model = Sequential()
    model.add(Dense(128, input_dim=i, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    # 3. 컴파일, 훈련
    model.compile(loss='sparse_categorical_crossentropy', 
                optimizer='adam', 
                metrics=['acc'],
                )

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_acc', 
                    mode='max',
                    patience=50,
                    verbose=1,   # stop이 어느 epoch에서 걸렸는지 출력해줌(modelcheckpoint도 적용가능)
                    restore_best_weights=True,
                    )

    start = time.time()
    hist = model.fit(x_train1, y_train, 
                    epochs=200, 
                    batch_size=64, 
                    verbose=1, 
                    validation_split=0.2,
                    callbacks=[es],
                    )
    end = time.time()

    # 4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=1)   # evaluation도 verbose 옵션사용가능
    print('loss :', loss[0])
    print('acc :', loss[1])

    y_pred = model.predict(x_test1)
    print(y_pred.shape) # (10000, 10)
    print(y_test.shape) # (10000, 10)

    y_pred = np.argmax(y_pred, axis=1)      # axis = 1 : 행 방향
    print(y_pred)           # [1 6 1 ... 1 1 6]
    print(type(y_test))     # <class 'numpy.ndarray'>
    #y_test = y_test.values  # 이미 y가 nparray이기 때문에 변환 불필요
    # y_test = np.argmax(y_test, axis=1)
    print(y_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"n_components가 {i}일때 걸린 시간 :", round(end-start, 2), "초")
    print(f"n_components가 {i}일때 accuracy : ", acc) 
    
    # 주성분축소 X
    # 걸린 시간 : 118.91 초
    # accuracy : 0.9815000295639038
    
    # n_components가 152일때 걸린 시간 : 143.51 초
    # n_components가 152일때 accuracy :  0.9768
    
    # n_components가 331일때 걸린 시간 : 140.51 초
    # n_components가 331일때 accuracy :  0.9758
    
    # n_components가 486일때 걸린 시간 : 148.48 초
    # n_components가 486일때 accuracy :  0.9757
    
    # n_components가 713일때 걸린 시간 : 155.72 초
    # n_components가 713일때 accuracy :  0.975
    
    # n_components가 784일때 걸린 시간 : 156.38 초
    # n_components가 784일때 accuracy :  0.9744
