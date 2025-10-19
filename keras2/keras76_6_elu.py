### <<56>>

import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(-5, 5, 0.1)

def elu(x, alpha):
    return (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))

# lambda : 일회용 함수
elu = lambda x, alpha : (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))

x = np.arange(-5, 5, 0.1)   # -5 부터 5 까지 0.1간격 배열 생성
print(x)
print(len(x))

y = elu(x, 10)

plt.plot(x, y)
plt.grid()
plt.show()