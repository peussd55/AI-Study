### <<63>>

import tensorflow as tf
import numpy as np
tf.random.set_random_seed(333)

# 1. 데이터
# x1_data = [73., 93., 89., 96., 73.]
# x2_data = [80., 88., 91., 98., 66.]
# x3_data = [75., 93., 90., 100., 70.]
# y_data = [152., 185., 180., 196., 142.]
x_data = np.array([[73, 80, 75],
                  [93, 88, 91],
                  [89, 91, 90],
                  [96, 98, 100],
                  [73, 66, 70]],
                  dtype=np.float32)

y_data = np.array([[152], [185], [180], [196], [142]], dtype=np.float32)

# w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
# w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
# w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
# b =  tf.compat.v1.Variable([0], dtype=tf.float32, name='bias')

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])   # Nx3
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])   # Nx1
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]), name='weights')    # 3X1
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='bias')   # (,1)

# 2. 모델구성
# hypothesis = x*w + b
hypothesis = tf.compat.v1.matmul(x, w) + b #행렬연산

# 3-1. 컴파일 및 훈련 준비
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3-2. 훈련 및 예측
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 10000
    for step in range(epochs):
        _, loss_val = sess.run(
            [train, loss],
            feed_dict={x: x_data, y: y_data}
        )
        if step % 100 == 0:
            print(f"Step: {step}, Loss: {loss_val}")

    print("============= predict =============")

    