### <<56>>

import numpy as np 
import matplotlib.pyplot as plt 

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# lambda : 일회용 함수
sigmoid = lambda x : 1 / (1 + np.exp(x))

x = np.arange(-5, 5, 0.1)   # -5 부터 5 까지 0.1간격 배열 생성
print(x)
print(len(x))

y = sigmoid(x)  # 레이어의 ouput을 0~1로 한정해서 연산폭발을 예방할 수 있다.

plt.plot(x, y)
plt.grid()
plt.show()