### <<63>>

import tensorflow as tf
import numpy as np
tf.compat.v1.random.set_random_seed(777)
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# 1. 데이터
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=333, 
)

print(x_train.shape, x_test.shape)  # (353, 10) (89, 10)
# y 데이터의 shape을 (N,)에서 (N, 1)로 변경
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(y_train.shape, y_test.shape)  # (353, 1) (89, 1)

# placeholder 정의
# y의 shape을 [None, 1]로 수정
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
print(x.shape, y.shape)             # (?, 10) (?, 1)

# 가중치(w)와 편향(b) 정의
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]), name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')
print(w.shape, b.shape)             # (10, 1) (1,)

# 2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b  # 행렬곱 (?, 10)x(10, 1) -> (?,1)

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
# r2_score : 0.4546133012099698
# mse : 2891.588361624194
