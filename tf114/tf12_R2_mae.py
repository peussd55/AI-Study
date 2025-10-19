### <<63>>

import tensorflow as tf
tf.random.set_random_seed(333)

import matplotlib.pyplot as plt

# 1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

x_test_data = [6,7,8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# 2. 모델구성
hypothesis = x * w + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

#################################### 옵티마이저 ####################################
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.02)
train = optimizer.minimize(loss)

# 3-2. 훈련
loss_val_list = []
w_val_list = []
y_pred = []
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 1001
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                         feed_dict={x:x_data, y:y_data})
        if step % 100 == 0:
            print(step, loss_val, w_val, b_val)
        
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
        
    print("============= predict =============")
    y_predict = x_test * w_val + b_val
    print("[6,7,8]의 결과", sess.run(y_predict, feed_dict={x_test:x_test_data}))
    # [6,7,8]의 결과 [13.013848 15.019644 17.02544 ]
    y_pred = sess.run(y_predict, feed_dict={x_test:x_test_data})
    
########### 실습 : R2, mae 만들기 #############
from sklearn.metrics import r2_score, mean_absolute_error
y_true = [13, 15, 17]
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
# 또는
y_prediction = x_test_data * w_val + b_val
r2 = r2_score(y_true, y_prediction)
mae = mean_absolute_error(y_true, y_prediction)

print('r2_score :', r2)
print('mae :', mae)
# r2_score : 0.9182005339571333
# mae : 0.45487213134765625