### <<62>>

import tensorflow as tf
tf.random.set_random_seed(333)

# 1. 데이터
# x = [1,2,3,4,5]
# y = [4,6,8,10,12]
x = tf.placeholder(tf.float32, shape=[None])    # None : 1차원 배열
y = tf.placeholder(tf.float32, shape=[None])    # None : 1차원 배열

# w = tf.Variable(111, dtype=tf.float32)    # 가중치(w)은 계속 업데이트되어야하는 값이기때문에 Variable로 할당
# b = tf.Variable(0, dtype=tf.float32)      # 편향(b)도 계속 업데이트되어야하는 값이기때문에 Variable로 할당

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)    # 정규분포(Normal Distribution)를 따르는 랜덤한 숫자 1개를 생성
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)    # 정규분포(Normal Distribution)를 따르는 랜덤한 숫자 1개를 생성
# ex) random_normal([2,2]) : 정규분포(Normal Distribution)를 따르는 랜덤한 숫자 4개를 생성 후 (2,2) shape 행렬 만듦.
print(w)    # <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>

# 2. 모델구성
# y = wx + b
hypothesis = x * w + b

# 3-1. 컴파일
# model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))                     # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)    # sgd
train = optimizer.minimize(loss)                # 역전파 및 가중치 업데이트

# 3-2. 훈련
# sess = tf.compat.v1.Session()        
with tf.compat.v1.Session() as sess:  # 종료 후 자동 close
    sess.run(tf.global_variables_initializer())

    # model.fit()
    epochs = 2000
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
                                             feed_dict={x:[1,2,3,4,5], y:[4,6,8,10,12]})
        
        if step % 20 == 0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)     # 1980 6.4233063e-12 [2.0000017] [1.9999943]
        
# sess.close()