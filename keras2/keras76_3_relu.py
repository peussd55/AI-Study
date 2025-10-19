### <<56>>

import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(-5, 5, 0.1)
# y = np.tanh(x)

# def relu(x):
#     return np.maximum(0, x)

# lambda : 일회용 함수
relu = lambda x : np.maximum(0, x)

x = np.arange(-5, 5, 0.1)   # -5 부터 5 까지 0.1간격 배열 생성
print(x)
print(len(x))

y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()