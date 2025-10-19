### <<62>>

# 07번을 카피해서 변수초기화 2번 (.eval(session=sess))으로 바꿔보기

import tensorflow as tf
tf.random.set_random_seed(333)

# 1. 데이터
x_data = [1,2,3,4,5]
y_data = [4,6,8,10,12]
x = tf.placeholder(tf.float32, shape=[None])    # None : 1차원 배열
y = tf.placeholder(tf.float32, shape=[None])    # None : 1차원 배열

x_test_data = [6,7,8]
#[실습] predict 뽑기 : [14, 16, 18]

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
        # 1) 학습 수행
        sess.run(train, feed_dict={x: x_data, y: y_data})
        
        # 2) loss, w, b 값을 .eval()로 가져오기
        if step % 20 == 0:
            loss_val = loss.eval(session=sess, feed_dict={x: x_data, y: y_data})
            w_val    = w.eval(session=sess)
            b_val    = b.eval(session=sess)
            print(step, loss_val, w_val, b_val)
            
    # 4. 예측
    # 4.1. placeholder 방식
    x_test = tf.placeholder(tf.float32, shape=[None])
    y_predict = x_test * w_val + b_val
    print("예측 결과1:", sess.run(y_predict, feed_dict={x_test: x_test_data}))
    # 예측 결과1: [14.000004 16.000006 18.000008]
    
    # 4.2. 파이썬(넘파이) 방식
    y_predict2 = x_test_data * w_val + b_val
    print("예측 결과2:", y_predict2)
    # 예측 결과2: [14.00000429 16.00000596 18.00000763]

    # 4.3. 
    prediction = sess.run(hypothesis, feed_dict={x: x_test_data})
    print("예측 결과3:", prediction)
    # 예측 결과3: [14.000004 16.000006 18.000008]

# sess.close()