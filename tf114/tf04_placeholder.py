### <<61>>

import tensorflow as tf
print(tf.__version__)   # 1.14.0

# node1 = tf.constant(3.0)
# node2 = tf.constant(3.0)
# node3 = node1 + node2

node1 = tf.compat.v1.placeholder(tf.float32)    # placeholder : 입력값공간할당 (상수 또는 변수)
node2 = tf.compat.v1.placeholder(tf.float32)    # placeholder : 입력값공간할당 (상수 또는 변수)
node3 = node1 + node2

sess = tf.compat.v1.Session()
# print(sess.run(node3))    # error
print(sess.run(node3, feed_dict={node1:3, node2:4}))            # 7.0
print(sess.run(node3, feed_dict={node1:10, node2:17}))          # 27.0

node3_triple = node3 * 3
print(node3_triple)                                             # Tensor("mul:0", dtype=float32)
print(sess.run(node3_triple, feed_dict={node1:3, node2:4}))     # 21.0

# tensor1은 계산구조(그래프)를 먼저 만들고 마지막에 값(feed_dict)을 넣는다.
