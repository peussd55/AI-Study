### <<61>>

import tensorflow as tf

# 1. 데이터
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111, dtype=tf.float32)    # 가중치(w)은 계속 업데이트되어야하는 값이기때문에 Variable로 할당
b = tf.Variable(0, dtype=tf.float32)      # 편향(b)도 계속 업데이트되어야하는 값이기때문에 Variable로 할당

# 2. 모델구성
# y = wx + b
hypothesis = x * w + b

# 3-1. 컴파일
# model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))                     # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)    # sgd
train = optimizer.minimize(loss)                # 역전파 및 가중치 업데이트

# 3-2. 훈련
sess = tf.compat.v1.Session()                   # 세션초기화
sess.run(tf.global_variables_initializer())     # 변수초기화

# model.fit()
epochs = 2000
for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))
        
sess.close()