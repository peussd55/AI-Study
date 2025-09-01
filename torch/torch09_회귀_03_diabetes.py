import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error # 회귀 스코어 임포트

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print("torch:", torch.__version__,'사용:', DEVICE)

seed=592

datasets = load_diabetes()
x=datasets.data
y=datasets.target 

print(x.shape) #(20640, 8)

x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    random_state=seed,
    test_size=0.2,
)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

