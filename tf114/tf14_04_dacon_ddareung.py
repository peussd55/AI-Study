### <<63>>

import tensorflow as tf
import numpy as np
import pandas as pd
tf.compat.v1.random.set_random_seed(777)
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

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
    x,y,
    test_size=0.25, 
    random_state=214
)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (1094, 9) (365, 9) (1094,) (365,)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

# placeholder 정의
# y의 shape을 [None, 1]로 수정
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 9])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
print(x.shape, y.shape)             # (?, 9) (?, 1)

# 가중치(w)와 편향(b) 정의
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([9,1]), name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')
print(w.shape, b.shape)             # (9, 1) (1,)

# 2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b  # 행렬곱 (?, 9)x(9, 1) -> (?,1)

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.02)
train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 10000
    for step in range(epochs):
        # feed_dict에 올바른 shape의 y_train 전달
        _, loss_val, w_val, b_val = sess.run(
            [train, loss, w, b],
            feed_dict={x: x_train, y: y_train}
        )
        if step % 1000 == 0:
            print(f"Step: {step}, Loss: {loss_val}")

    print("============= predict =============")
    
    # 예측은 훈련 때 정의한 hypothesis를 그대로 사용
    # feed_dict를 통해 x placeholder에 테스트 데이터를 전달
    # y_predict = tf.compat.v1.matmul(x_test, w_val) + b_val      # float32, float64 타입 안맞아서 오류. cast해줘야함.
    y_predict = sess.run(hypothesis, feed_dict={x: x_test})       # float64 타입인 x_test를 tf.float32로 placeholder된 x에 넣어서 cast됐음
    # run : 그래프 계산을 실행

# 4. 평가
# 세션이 끝난 후, Numpy 배열인 y_test와 y_predict를 사용하여 평가
r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)

print('r2_score :', r2)
print('mse :', mse)
# r2_score : 0.5936682553176267
# mse : 2875.2499556565567
