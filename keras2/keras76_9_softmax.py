### <<56>>

import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(1, 5) # 1 부터 5 까지 1간격 배열 생성

# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x))

# lambda : 일회용 함수
softmax = lambda x : np.exp(x) / np.sum(np.exp(x))

print(x)
print(len(x))

y = softmax(x)
ratio = y
labels = y
plt.pie(ratio, labels, shadow=True, startangle=90)
plt.grid()
plt.show()