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

rate = tf.compat.v1.placeholder(tf.float32)

# 2. 모델 (MLP 구조로 변경)
# -----------------------------------------------------------------
# Layer 1: Hidden Layer (은닉층)
# 입력 특성(feature)이 2개, 히든 노드의 개수를 3개로 설정
w1 = tf.compat.v1.Variable(tf.random_normal([2, 3], name='weights1'))
b1 = tf.compat.v1.Variable(tf.zeros([3], name='bias1'))
# (N, 2) * (2, 3) -> (N, 3)
################ relu 적용 ################
# layer1 = tf.matmul(x, w1) + b1
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)   
# layer1 = tf.nn.selu(tf.matmul(x, w1) + b1)  
# layer1 = tf.nn.leaky_relu(tf.matmul(x, w1) + b1)               
# layer1 = tf.nn.elu(tf.matmul(x, w1) + b1)
#############################################


# Layer 2: Hidden Layer (은닉층)
# Layer 1의 출력(3개)을 입력으로 받아 최종 결과(4개)를 출력
w2 = tf.compat.v1.Variable(tf.random_normal([3, 4], name='weights2'))
b2 = tf.compat.v1.Variable(tf.zeros([4], name='bias2'))
layer2 = tf.compat.v1.sigmoid(tf.matmul(layer1, w2) + b2)
# (N, 3) * (3, 4) -> (N, 4)
################ dropout 적용 ################
layer2 = tf.nn.dropout(layer2, rate=rate)
# layer2 = tf.nn.dropout(layer2, keep_prob=0.8)
#############################################

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

# 평가 지표 정의
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 3-2. 훈련
epochs = 5001
for step in range(epochs):
    # sess.run에 accuracy를 추가하여 훈련 과정에서 정확도를 함께 계산
    cost_val, acc_val, _ = sess.run([loss, accuracy, train], feed_dict={x: x_data, y: y_data, rate: 0.2})
    
    if step % 500 == 0:
        # loss와 함께 accuracy(acc_val)를 출력
        print(f"Step: {step}, Loss: {cost_val}, Accuracy: {acc_val}")

# 4-1. 평가
print("\n==== 최종 평가 ====")
# 최종 평가 시에는 이미 훈련된 가중치를 사용하므로 train op은 실행할 필요가 없음
h, p, a = sess.run([hypothesis, predicted, accuracy],
                   feed_dict={x: x_data, y: y_data, rate: 0.0})

print("Hypothesis (예측값) :\n", h)
print("Predicted (0 또는 1) :\n", p)
print("Accuracy (정확도) :", a)
# ==== 최종 평가 ====
# Hypothesis (예측값) : datatype : Tensor
#  [[0.12889601]
#  [0.9997929 ]
#  [0.9924495 ]
#  [0.1314389 ]]
# Predicted (0 또는 1) : datatype : Numpy
#  [[0.]
#  [1.]
#  [1.]
#  [0.]]

sess.close()

# 4-2. 예측
from sklearn.metrics import accuracy_score
import numpy as np

final_acc = accuracy_score((h > 0.5).astype(int), y_data)
# 또는 
final_acc = accuracy_score(p, y_data)
print('최종 acc :', final_acc)  # 최종 acc : 1.0
