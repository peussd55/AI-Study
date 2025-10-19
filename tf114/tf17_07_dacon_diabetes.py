### <<64>>

import tensorflow as tf
import pandas as pd
import numpy as np
tf.compat.v1.random.set_random_seed(777)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터
path = './_data/dacon/diabetes/'      

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   
print(train_csv)                    # [652 rows x 9 columns]             

test_csv = pd.read_csv(path + 'test.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(test_csv)                     # [116 rows x 8 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(submission_csv)               # [116 rows x 1 columns] 

print(train_csv.shape)              #(652,9)
print(test_csv.shape)               #(116,8)
print(submission_csv.shape)         #(116,1)

print("######################################################################################################################")                                              
print(train_csv.columns)  # Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                          # 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
                          # dtype='object')
print("######################################################################################################################")                                              
print(train_csv.info())             
print("######################################################################################################################")                          
print(train_csv.describe())

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome'] 
x = x.replace(0, np.nan)                
x = x.fillna(x.mean())
test_csv = test_csv.fillna(x.mean())

print(train_csv.isna().sum())
print(test_csv.isna().sum())

print(x) #652 rows x 8 
print(y) #652

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=813, shuffle=True, stratify=y
    )
print(x_train.shape, x_test.shape)  #(521, 8) (131, 8)
print(y_train.shape, y_test.shape)  #(521,)  (131,)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 

# placeholder 정의
# y의 shape을 [None, 1]로 수정
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
print(x.shape, y.shape)             # (?, 8) (?, 1)

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1], name='weights', dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias', dtype=tf.float32))

# 2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
# logits = tf.compat.v1.matmul(x, w) + b 
# hypothesis = tf.compat.v1.sigmoid(logits) # 예측 시에는 그대로 사용

# 3-1. 컴파일
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

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
# 최종 weights : [[ 0.49829665]
#  [ 1.0812762 ]
#  [ 0.01758089]
#  [ 0.04895046]
#  [-0.00488793]
#  [ 0.50926673]
#  [ 0.27449122]
#  [ 0.11293481]] 최종 bias : [-0.8743126] 최종 acc : 0.7633588

sess.close()

from sklearn.metrics import accuracy_score
final_acc = accuracy_score(final_pred, y_test)
print('최종 acc :', final_acc) 
# 최종 acc : 0.7633587786259542