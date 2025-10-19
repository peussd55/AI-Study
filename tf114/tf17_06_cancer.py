### <<64>>

import tensorflow as tf
tf.compat.v1.random.set_random_seed(777)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=813, shuffle=True, stratify=y
)

print(x_train.shape, x_test.shape)  # (455, 30) (114, 30)
# y 데이터의 shape을 (N,)에서 (N, 1)로 변경
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(y_train.shape, y_test.shape)  # (455, 1) (114, 1)

# placeholder 정의
# y의 shape을 [None, 1]로 수정
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
print(x.shape, y.shape)             # (?, 30) (?, 1)

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1], name='weights', dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias', dtype=tf.float32))

scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 

# 2. 모델
# hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
logits = tf.compat.v1.matmul(x, w) + b 
hypothesis = tf.compat.v1.sigmoid(logits) # 예측 시에는 그대로 사용

# 3-1. 컴파일
# loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))         # binary_crossentropy / 극단값때문에 이거쓰면 Nan뜸
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# [실습] acc 넣기

# 3-2. 훈련
epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_train, y:y_train})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)

# 4. 평가, 예측        
y_predict = sess.run(hypothesis, feed_dict={x: x_test})
prediction = tf.cast(y_predict > 0.5, tf.float32)
final_pred = sess.run(prediction)
acc = tf.reduce_mean(tf.cast(tf.equal(final_pred, y_test), dtype=tf.float32))
print(acc)                      # Tensor("Mean_1:0", shape=(), dtype=float32)

print('최종 weights :', w_val, '최종 bias :', b_val, '최종 acc :', sess.run(acc))
# 최종 weights : [[-0.5121045]
#  ... 최종 bias : [0.66338205] 최종 acc : 0.9649123

sess.close()

from sklearn.metrics import accuracy_score
final_acc = accuracy_score(final_pred, y_test)
print('최종 acc :', final_acc)  # 최종 acc : 0.9649123
# 최종 acc :  0.9649122807017544