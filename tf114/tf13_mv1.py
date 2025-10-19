### <<63>>

import tensorflow as tf
import numpy as np
tf.random.set_random_seed(333)

# 1. 데이터
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
b =  tf.compat.v1.Variable([0], dtype=tf.float32, name='bias')

x1 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x3 = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

x_test_data = [60.,70.,80.,90.,100.]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# 2. 모델구성
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

# 3-1. 컴파일 및 훈련 준비
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# ==================== 핵심 수정 부분 ====================
# 예측용 입력 데이터 'x_test_data'를 (샘플 수, 특성 수) 형태의 2D 배열로 정의합니다.
# 이 변수 하나에 모든 테스트 특성 정보가 담겨 있습니다.
x_test_data = np.array([60., 70., 80., 90., 100.], dtype=np.float32)

# 예측 모델의 실제 정답 값(y_true)도 미리 정의합니다.
# 실제 프로젝트에서는 검증/테스트 데이터셋의 라벨을 사용합니다.

# =======================================================

# 3-2. 훈련 및 예측
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 10000
    for step in range(epochs):
        _, loss_val = sess.run(
            [train, loss],
            feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data}
        )
        if step % 100 == 0:
            print(f"Step: {step}, Loss: {loss_val}")

    print("============= predict =============")
    