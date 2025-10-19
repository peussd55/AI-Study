### <<65>>

import tensorflow as tf
tf.compat.v1.random.set_random_seed(777)

# 1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]   # (2,2)
y_data = [[0], [1], [1], [0]]           # (2,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# w = tf.compat.v1.Variable(tf.random_normal([2,1], name='weights'))
# b = tf.compat.v1.Variable(tf.zeros([1], name='bias'))

# 2. 모델 (MLP 구조로 변경)
# -----------------------------------------------------------------
# Layer 1: Hidden Layer (은닉층)
# 입력 특성(feature)이 2개, 히든 노드의 개수를 3개로 설정
w1 = tf.compat.v1.Variable(tf.random_normal([2, 3], name='weights1'))
b1 = tf.compat.v1.Variable(tf.zeros([3], name='bias1'))
layer1 = tf.matmul(x, w1) + b1                                  
# (N, 2) * (2, 3) -> (N, 3)


# Layer 2: Hidden Layer (은닉층)
# Layer 1의 출력(3개)을 입력으로 받아 최종 결과(4개)를 출력
w2 = tf.compat.v1.Variable(tf.random_normal([3, 4], name='weights2'))
b2 = tf.compat.v1.Variable(tf.zeros([4], name='bias2'))
layer2 = tf.compat.v1.sigmoid(tf.matmul(layer1, w2) + b2)
# (N, 3) * (3, 4) -> (N, 4)

# Layer 3: Output Layer (출력층)
# Layer 2의 출력(4개)을 입력으로 받아 최종 결과(1개)를 출력
w3 = tf.compat.v1.Variable(tf.random_normal([4, 1], name='weights2'))
b3 = tf.compat.v1.Variable(tf.zeros([1], name='bias2'))
hypothesis = tf.compat.v1.sigmoid(tf.matmul(layer2, w3) + b3)   
# (N, 4) * (4, 1) -> (N, 1)

"""
🌏 [중요] 모델의 비선형성 부여는 은닉층의 활성화함수에 의해서 결정된다. 출력층에서의 활성화함수는 모델의 비선형성과는 아무사 상관이없다.
-> 히든 레이어 중에 한 곳이라도 sigmoid로 감싸서 비선형성을 줘야 acc가 1.0달성한다.
 (1) : 출력층 에서의 활성화함수(sigmoid) : 최종 결과값을 확률처럼 해석하기 위해 사용.
 (2) : 은닉층 에서의 활성화함수(sigmoid) : 레이어에 비선형성을 부여. 
"""
# -----------------------------------------------------------------

# 3-1. 컴파일
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

# 학습률(learning_rate)을 조정하여 더 빠르게 수렴하도록 변경
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 3-2. 훈련
epochs = 5001
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={x: x_data, y: y_data})
    if step % 500 == 0:
        print(step, 'loss :', cost_val)

# 4. 평가, 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# 훈련된 모델로 예측 및 정확도 계산
h, p, a = sess.run([hypothesis, predicted, accuracy],
                   feed_dict={x: x_data, y: y_data})

print("\n==== 최종 결과 ====")
print("Hypothesis (예측값) :\n", h)
print("Predicted (0 또는 1) :\n", p)
print("Accuracy (정확도) :", a)
# ==== 최종 결과 ====
# Hypothesis (예측값) :
#  [[0.01365356]
#  [0.9782511 ]
#  [0.98166394]
#  [0.02548812]]
# Predicted (0 또는 1) :
#  [[0.]
#  [1.]
#  [1.]
#  [0.]]
# Accuracy (정확도) : 1.0