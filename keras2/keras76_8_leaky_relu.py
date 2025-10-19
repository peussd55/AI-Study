### <<56>>

import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(-5, 5, 0.1)
alpha = 0.01
def leaky_relu(x, alpha):
    # return np.maximum(alpha*x, x)
    return np.where(x > 0, x, alpha * x)

# lambda : 일회용 함수
leaky_relu = lambda x, alpha : np.maximum(alpha*x, x)

x = np.arange(-5, 5, 0.1)   # -5 부터 5 까지 0.1간격 배열 생성
print(x)
print(len(x))

y = leaky_relu(x, alpha)

plt.plot(x, y)
plt.grid()
plt.show()