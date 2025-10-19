### <<61>>

import tensorflow as tf
sess = tf.compat.v1.Session()

a = tf.Variable([2], dtype=tf.float32)  # 변수에 초기값 2 할당
b = tf.Variable([1], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer()  # 변수초기화
sess.run(init)

print(sess.run(a + b))  # [3.]