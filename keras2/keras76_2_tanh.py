### <<56>>

import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(-5, 5, 0.1)
# y = np.tanh(x)

# lambda : 일회용 함수
tanh = lambda x : (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

x = np.arange(-5, 5, 0.1)   # -5 부터 5 까지 0.1간격 배열 생성
print(x)
print(len(x))

y = tanh(x)  # 레이어의 ouput을 -1~1로 한정해서 연산폭발을 예방할 수 있다.

plt.plot(x, y)
plt.grid()
plt.show()