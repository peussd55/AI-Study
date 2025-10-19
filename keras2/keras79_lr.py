### <<56>>
import numpy as np 
import matplotlib.pyplot as plt 

x = 10
y = 100
w = 0.001
lr = 0.001
epochs = 1000

for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y) **2 # 바이어스, 시그마 고려X
    
    print('Loss :', round(loss, 4), '\tPredict :', round(hypothesis, 4))
    
    up_predict = x*(w + lr)
    up_loss = (y - up_predict) ** 2
    
    down_predict = x*(w - lr)
    down_loss = (y - down_predict) ** 2
    
    if(up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr