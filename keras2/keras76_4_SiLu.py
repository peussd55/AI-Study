### <<56>>

# 2017 구글 swish 발표

import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(-5, 5, 0.1)

def silu(x):    # = swish 함수
    return x * (1 / (1 + np.exp(-x)))


# lambda : 일회용 함수
silu = lambda x : x * (1 / (1 + np.exp(-x)))

x = np.arange(-5, 5, 0.1)   # -5 부터 5 까지 0.1간격 배열 생성
print(x)
print(len(x))

y = silu(x)

plt.plot(x, y)
plt.grid()
plt.show()