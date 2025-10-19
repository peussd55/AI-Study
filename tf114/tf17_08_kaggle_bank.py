### <<64>>

import tensorflow as tf
import pandas as pd
import numpy as np
tf.compat.v1.random.set_random_seed(777)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. 데이터
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 결측치 확인
print(train_csv.isna().sum())   # 결측치 없음
print(test_csv.isna().sum())   # 결측치 없음

# 문자 데이터 수치화(인코딩)
le_geo = LabelEncoder()
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

# CustomerId, Surname 제거
train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=8282, shuffle=True, stratify=y
)

print(x_train.shape, x_test.shape)  # (115523, 10) (49511, 10)
print(y_train.shape, y_test.shape)  # (115523,) (49511,)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 

# placeholder 정의
# y의 shape을 [None, 1]로 수정
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
print(x.shape, y.shape)             # (?, 10) (?, 1)

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1], name='weights', dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias', dtype=tf.float32))

# 2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
# logits = tf.compat.v1.matmul(x, w) + b 
# hypothesis = tf.compat.v1.sigmoid(logits) # 예측 시에는 그대로 사용

# 3-1. 컴파일
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# [실습] acc 넣기

# 3-2. 훈련
epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_train, y:y_train})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)

# 4. 평가, 예측        
y_predict = sess.run(hypothesis, feed_dict={x: x_test})
prediction = tf.cast(y_predict > 0.5, tf.float32)
final_pred = sess.run(prediction)
acc = tf.reduce_mean(tf.cast(tf.equal(final_pred, y_test), dtype=tf.float32))
print(acc)                      # Tensor("Mean_1:0", shape=(), dtype=float32)

print('최종 weights :', w_val, '최종 bias :', b_val, '최종 acc :', sess.run(acc))
# 최종 weights : [[-0.06246765]
#  [ 0.09076148]
#  [-0.33284205]
#  [ 0.8496483 ]
#  [-0.05017648]
#  [ 0.14809522]
#  [-0.4533342 ]
#  [-0.06512568]
#  [-0.6383241 ]
#  [ 0.0531674 ]] 최종 bias : [-1.7291217] 최종 acc : 0.8272505

sess.close()

from sklearn.metrics import accuracy_score
final_acc = accuracy_score(final_pred, y_test)
print('최종 acc :', final_acc) 
# 최종 acc : 0.8272505099876795