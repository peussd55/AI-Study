### <<64>>

import tensorflow as tf
import numpy as np
tf.compat.v1.random.set_random_seed(777)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터
x_data = [
    [1,2,1,1],
    [2,1,3,2],
    [3,1,3,4],
    [4,1,5,5],
    [1,7,5,5],
    [1,2,5,6],
    [1,6,6,6],
    [1,7,6,7],
]
y_data = [
    [0,0,1],
    [0,0,1],
    [0,0,1],
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [1,0,0],
    [1,0,0],
]
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.compat.v1.Variable(tf.random_normal([4,3], name='weights', dtype=tf.float32))  
b = tf.compat.v1.Variable(tf.zeros([3], name='bias', dtype=tf.float32))
# x * w : (N, 4) * (4, 3) => (N,3)
# w * x 와는 연산이 완전히 다르다.

# 2. 모델
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

predicted = tf.argmax(hypothesis, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y, 1)), dtype=tf.float32))

# 3-2. 훈련
epochs = 5000
for step in range(epochs):
    cost_val, _, w_val, b_val, acc_val = sess.run([loss, train, w, b, accuracy], feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss :', cost_val, 'accuracy :', acc_val)

# 4. 평가, 예측        
y_predict = sess.run(hypothesis, feed_dict={x: x_data})
y_predict = np.argmax(y_predict, 1)
print(y_predict)    # [2 2 2 1 1 1 0 0]
sess.close()

from sklearn.metrics import accuracy_score
final_acc = accuracy_score(y_predict, np.argmax(y_data, 1))
print('최종 acc :', final_acc)  # 최종 acc : 1.0
