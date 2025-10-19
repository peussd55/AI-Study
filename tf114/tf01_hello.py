### <<61>>
import tensorflow as tf
print(tf.__version__)               
# 1.14.0

print("hello world")

hello = tf.constant("hello world")
print(hello)                        
# Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
print(sess.run(hello))    # run : 그래프 연산을 실행시킴          
# b'hello world'

# 텐서1은 모든 것을 Session에 넣어서 넣는다 -> 텐서머신
# Session.run -> 머신의 결과를 출력