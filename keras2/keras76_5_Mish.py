### <<56>>

import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(-5, 5, 0.1)

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

# lambda : 일회용 함수
mish = lambda x : x * np.tanh(np.log(1 + np.exp(x)))

x = np.arange(-5, 5, 0.1)   # -5 부터 5 까지 0.1간격 배열 생성
print(x)
print(len(x))

y = mish(x)

plt.plot(x, y)
plt.grid()
plt.show()