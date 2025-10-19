### <<64>>

import tensorflow as tf
tf.compat.v1.random.set_random_seed(777)

# 1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1], name='weights', dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias', dtype=tf.float32))

# 2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)

# 3-1. 컴파일
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis)) # binary_crossentropy

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# [실습] acc 넣기

# 3-2. 훈련
epochs = 404
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)

# 4. 평가, 예측        
y_predict = sess.run(hypothesis, feed_dict={x: x_data})
prediction = tf.cast(y_predict > 0.5, tf.float32)
final_pred = sess.run(prediction)
acc = tf.reduce_mean(tf.cast(tf.equal(final_pred, y_data), dtype=tf.float32))
print(acc)                      # Tensor("Mean_1:0", shape=(), dtype=float32)

print('최종 weights :', w_val, '최종 bias :', b_val, '최종 acc :', sess.run(acc))
# 최종 weights : [[1.9226679]
#  [0.706727 ]] 최종 bias : [-8.058993] acc : 1.0

sess.close()

from sklearn.metrics import accuracy_score
final_acc = accuracy_score(final_pred, y_data)
print('최종 acc :', final_acc)  # 최종 acc : 1.0