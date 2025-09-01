import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
seed = 2497
random.seed(seed)
np.random.seed(seed)
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

###### prepare data
    ### load data
from sklearn.datasets import load_diabetes
x, y = load_diabetes(return_X_y=True)

    ### train test split
from sklearn.model_selection import train_test_split
x_tr, x_ts, y_tr, y_ts = train_test_split(x,y, train_size=0.8,
                                          shuffle=True, random_state=seed)

    ### scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_tr = sc.fit_transform(x_tr)
x_ts = sc.transform(x_ts)

    ### transform to tensor
x_tr = torch.tensor(x_tr, dtype=torch.float32).to(DEVICE)
x_ts = torch.tensor(x_ts, dtype=torch.float32).to(DEVICE)
y_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(DEVICE)

###### learn
    ### model
model = nn.Sequential(
    nn.Linear(10,32),
    nn.ReLU(),
    nn.Linear(32,64),
    nn.ReLU(),
    nn.Linear(64,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.ReLU(),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
def train(model, crit_, optim_, x, y):
    optim_.zero_grad()
    hyp_ = model(x)
    loss = crit_(hyp_, y)
    loss.backward()
    optim_.step()
    return loss.item()

    ### fit
epochs = 10000
for epoch in range(1,epochs+1):
    loss = train(model, criterion, optimizer, x_tr,y_tr)
    if epoch%100==0:print('epoch_{} Loss : {}'.format(epoch, loss))
    else:continue

###### eval / pred
    ### eval
def evaluate(model, crit_, x, y):
    model.eval()
    with torch.no_grad():
        y_cal = model(x)
        loss = crit_(y_cal,y)
        return loss.item()
    
print(evaluate(model, criterion, x_tr, y_tr))

    ### pred
from sklearn.metrics import mean_squared_error
y_pred = model(x_ts)
rmse = mean_squared_error(y_ts,y_pred.tolist())
print(rmse)
