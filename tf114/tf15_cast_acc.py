### <<64>>

import tensorflow as tf
aaa = tf.constant([0.3, 0.4, 0.8, 0.9])
bbb = tf.constant([0., 1., 1., 1.])

sess = tf.compat.v1.Session()

pred = tf.cast(aaa > 0.5, dtype=tf.float32)     # True => 1 // False => 0
# print(pred)               # Tensor("Cast:0", shape=(4,), dtype=float32)
predict = sess.run(pred)
print(predict)              # [0. 0. 1. 1.]

acc = tf.reduce_mean(tf.cast(tf.equal(predict, bbb), dtype=tf.float32))     # reduce_mean : 평균
print(sess.run(acc))        # 0.75

sess.close()