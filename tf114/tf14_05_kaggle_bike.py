### <<63>>

import tensorflow as tf
import numpy as np
import pandas as pd
tf.compat.v1.random.set_random_seed(777)
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# 1. 데이터
path = './_data/kaggle/bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [10886 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) # [6493 rows x 8 columns]

submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
print(submission_csv)   # [6493 rows x 1 columns]

print(train_csv.shape, test_csv.shape, submission_csv.shape)    # (10886, 11) (6493, 8) (6493, 1)

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object')
print(train_csv.info())
# <class 'pandas.core.frame.DataFrame'>
# Index: 10886 entries, 2011-01-01 00:00:00 to 2012-12-19 23:00:00
# Data columns (total 11 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   season      10886 non-null  int64
#  1   holiday     10886 non-null  int64
#  2   workingday  10886 non-null  int64
#  3   weather     10886 non-null  int64
#  4   temp        10886 non-null  float64
#  5   atemp       10886 non-null  float64
#  6   humidity    10886 non-null  int64
#  7   windspeed   10886 non-null  float64
#  8   casual      10886 non-null  int64
#  9   registered  10886 non-null  int64
#  10  count       10886 non-null  int64
print(train_csv.describe())
# dtypes: float64(3), int64(8)
# memory usage: 1020.6+ KB
# None
#              season       holiday    workingday       weather         temp         atemp      humidity     windspeed        casual    registered         count
# count  10886.000000  10886.000000  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000
# mean       2.506614      0.028569      0.680875      1.418427     20.23086     23.655084     61.886460     12.799395     36.021955    155.552177    191.574132
# std        1.116174      0.166599      0.466159      0.633839      7.79159      8.474601     19.245033      8.164537     49.960477    151.039033    181.144454
# min        1.000000      0.000000      0.000000      1.000000      0.82000      0.760000      0.000000      0.000000      0.000000      0.000000      1.000000
# 25%        2.000000      0.000000      0.000000      1.000000     13.94000     16.665000     47.000000      7.001500      4.000000     36.000000     42.000000
# 50%        3.000000      0.000000      1.000000      1.000000     20.50000     24.240000     62.000000     12.998000     17.000000    118.000000    145.000000
# 75%        4.000000      0.000000      1.000000      2.000000     26.24000     31.060000     77.000000     16.997900     49.000000    222.000000    284.000000
# max        4.000000      1.000000      1.000000      4.000000     41.00000     45.455000    100.000000     56.996900    367.000000    886.000000    977.000000
print(train_csv.isna().sum())
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
# casual        0
# registered    0
# count         0
print(test_csv.isna().sum())
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0

x = train_csv.drop(['count','casual','registered'], axis=1)
print(x)    # [10886 rows x 10 columns]

y = train_csv['count']
print(x.shape, y.shape) # (10886, 10) (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=777
)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (9797, 8) (1089, 8) (9797,) (1089,)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

# placeholder 정의
# y의 shape을 [None, 1]로 수정
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
print(x.shape, y.shape)             # (?, 8) (?, 1)

# 가중치(w)와 편향(b) 정의
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1]), name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')
print(w.shape, b.shape)             # (8, 1) (1,)

# 2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b  # 행렬곱 (?, 8)x(8, 1) -> (?,1)

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
# r2_score : 0.2557746310917024
# mse : 25321.956212316054
