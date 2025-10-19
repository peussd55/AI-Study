### <<64>>

import tensorflow as tf
import numpy as np
import pandas as pd
tf.compat.v1.random.set_random_seed(777)
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터
# 불러오기
datasets = fetch_covtype()
x_data = datasets.data
y_data = datasets.target
print(np.unique(y_data, return_counts=True))     # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# 원핫인코딩
y_data = y_data.reshape(-1, 1)
encoder = OneHotEncoder(sparse=0)
y_data = encoder.fit_transform(y_data)
print(x_data.shape, y_data.shape)            # (1797, 64) (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x_data,y_data,
    test_size=0.2,
    random_state=813, shuffle=True, stratify=y_data
)

scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 

# placeholder 정의
# y의 shape을 [None, 1]로 수정
x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_data.shape[1]])
print(x.shape, y.shape)             # (581012, 54) (581012, 7)

w = tf.compat.v1.Variable(tf.random_normal([x_data.shape[1], y_data.shape[1]], name='weights', dtype=tf.float32))  
b = tf.compat.v1.Variable(tf.zeros([y_data.shape[1]], name='bias', dtype=tf.float32))
# x * w : (N, 54) * (54, 7) => (N,7)

# 2. 모델
# hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
logits = tf.matmul(x, w) + b
hypothesis = tf.nn.softmax(logits)

# 3-1. 컴파일
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

predicted = tf.argmax(hypothesis, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y, 1)), dtype=tf.float32))

# 3-2. 훈련
epochs = 5000
for step in range(epochs):
    cost_val, _, w_val, b_val, acc_val = sess.run([loss, train, w, b, accuracy], feed_dict={x:x_train, y:y_train})
    if step % 20 == 0:
        print(step, 'loss :', cost_val, 'accuracy :', acc_val)

# 4. 평가, 예측        
y_predict = sess.run(hypothesis, feed_dict={x: x_test})
y_predict = np.argmax(y_predict, 1)
print(y_predict)
sess.close()

from sklearn.metrics import accuracy_score
final_acc = accuracy_score(y_predict, np.argmax(y_test, 1))
print('최종 acc :', final_acc)  # 최종 acc : 0.7182086521002039
