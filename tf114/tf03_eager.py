### <<61>>

import tensorflow as tf
print('tf version : ', tf.__version__)
print('즉시실행모드 : ', tf.executing_eagerly())
# tf114cpu
# tf version :  1.14.0 
# 즉시실행모드 :  False

# 가상환경 변경 tf274cpu
# tf version :  2.7.4 
# 즉시실행모드 :  True

tf.compat.v1.disable_eager_execution()
print('즉시실행모드 : ', tf.executing_eagerly())
# 즉시실행모드 :  False (tf274cpu)

# tf.compat.v1.enable_eager_execution()
# print('즉시실행모드 : ', tf.executing_eagerly())
# 즉시실행모드 :  True (tf274cpu)

hello = tf.constant("Hello World!")
sess = tf.compat.v1.Session()

print(sess.run(hello))  
# RuntimeError : 즉시실행모드가 켜져있을때
# b'Hello World!' : 즉시실행모드가 꺼져있을때
# -> 텐서플로2에서 텐서플로1 코드를 실행해야하는 경우가있을때 즉시실행모드로 실행하게할 수 있다.

#############################################################
# 즉시 실행모드 -> 텐서1의 그래프 형태의 구성 없이 자연스러운 파이썬 문법으로 실행
# tf.compat.v1.disable_eager_execution()    # 즉시 실행 모드 끄기 // 텐서플로우 1.0 문법(디폴트)
# tf.compat.v1.enable_eager_execution()     # 즉시 실행 모드 킴 // 텐서플로우 2.0 사용 가능

# sess.run() 실행시점
#   [가상환경]    [즉시실행모드]        [사용가능]
#   1.14.0      disable(디폴트)     b'Hello World!'
#   1.14.0      enable              error
#   2.7.4       disable(디폴트)     b'Hello World!'
#   2.7.4       enable              error

"""
Tensor1 은 '그래프 연산' 모드
Tensor2 는 '즉시 실행' 모드

tf.compat.v1.enable_eager_execution()    # 즉시 실행 모드 킴
-> Tensor2의 디폴트

tf.compat.v1.disable_eager_execution()   # 즉시 실행 모드 끄기
-> 그래프 연산모드로 돌아감
-> Tensor1코드를 쓸 수 있음

executing_eagerly()
-> True : 즉시 실행모드. Tensor 2 코드만 써야함
-> False : 그래프 연산모드. Tensor 1 코드를 쓸 수있음

"""