### <<65>>

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
tf.compat.v1.random.set_random_seed(777)

# 1. 데이터
datasets = load_iris()
x_data = datasets.data
y_data = datasets.target
print(np.unique(y_data, return_counts=True))
# (array([0, 1, 2]), array([50, 50, 50], dtype=int64))

# 원핫인코딩
y_data = y_data.reshape(-1, 1)
encoder = OneHotEncoder(sparse=0)
y_data = encoder.fit_transform(y_data)
print(x_data.shape, y_data.shape)            # (150, 4) (150, 3)
x_feature_num = x_data.shape[1]
y_feature_num = y_data.shape[1]

x_train, x_test, y_train, y_test = train_test_split(
    x_data,y_data,
    test_size=0.2,
    random_state=813, shuffle=True, stratify=y_data
)

scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 

x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_feature_num])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, y_feature_num])
rate_1 = tf.compat.v1.placeholder(tf.float32)
rate_2 = tf.compat.v1.placeholder(tf.float32)

# 2. 모델 (MLP 구조로 변경)
# -----------------------------------------------------------------
# Layer 1: Hidden Layer (은닉층)
w1 = tf.compat.v1.Variable(tf.random_normal([x_feature_num, 32], name='weights1'))
b1 = tf.compat.v1.Variable(tf.zeros([32], name='bias1'))
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)   
layer1 = tf.nn.dropout(layer1, rate=rate_1)

# Layer 2: Hidden Layer (은닉층)
w2 = tf.compat.v1.Variable(tf.random_normal([32, 16], name='weights2'))
b2 = tf.compat.v1.Variable(tf.zeros([16], name='bias2'))
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
layer2 = tf.nn.dropout(layer2, rate=rate_2)

# Layer 3: Hidden Layer (은닉층)
w3 = tf.compat.v1.Variable(tf.random_normal([16, 8], name='weights3'))
b3 = tf.compat.v1.Variable(tf.zeros([8], name='bias3'))
layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)

# Layer 4: Output Layer (출력층)
w4 = tf.compat.v1.Variable(tf.random_normal([8, y_feature_num], name='weights4'))
b4 = tf.compat.v1.Variable(tf.zeros([y_feature_num], name='bias4'))
logits = tf.matmul(layer3, w4) + b4
hypothesis = tf.nn.softmax(logits)  
# -----------------------------------------------------------------

# 3-1. 컴파일
# cost함수 정의
# [이진]
# loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
# [다중]
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))    # 이거 쓰면 0.3 나옴. 레이어 중에 하나라도 sigmoid 해야 0.9나옴
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# optimizer, train 정의
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 평가 지표 정의
# [이진]
# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
# [다중]
predicted = tf.argmax(hypothesis, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y, 1)), dtype=tf.float32))

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 3-2. 훈련
epochs = 5001
for step in range(epochs):
    # sess.run에 accuracy를 추가하여 훈련 과정에서 정확도를 함께 계산
    cost_val, acc_val, _ = sess.run([loss, accuracy, train], feed_dict={x: x_train, y: y_train, rate_1: 0.2, rate_2:0.1})
    
    if step % 500 == 0:
        # loss와 함께 accuracy(acc_val)를 출력
        print(f"Step: {step}, Loss: {cost_val}, Accuracy: {acc_val}")

# 4-1. 평가
print("\n==== 최종 평가 ====")
# 최종 평가 시에는 이미 훈련된 가중치를 사용하므로 train op은 실행할 필요가 없음
h, p, a = sess.run([hypothesis, predicted, accuracy],
                   feed_dict={x: x_test, y: y_test, rate_1: 0.0, rate_2:0.0})

print("Hypothesis (예측값) :\n", h)
print("Predicted (0 또는 1) :\n", p)
print("Accuracy (정확도) :", a)

sess.close()

# 4-2. 예측
from sklearn.metrics import accuracy_score
import numpy as np

# final_acc = accuracy_score(np.argmax(h, 1), y_test)
# 또는 
final_acc = accuracy_score(p, y_test)
print('최종 acc :', final_acc)
