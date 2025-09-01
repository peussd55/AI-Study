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
from sklearn.datasets import fetch_california_housing
x, y = fetch_california_housing(return_X_y=True)

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
# model = nn.Sequential(
#     nn.Linear(8,32),
#     nn.ReLU(),
#     nn.Linear(32,64),
#     nn.ReLU(),
#     nn.Linear(64,128),
#     nn.ReLU(),
#     nn.Linear(128,512),
#     nn.ReLU(),
#     nn.Linear(512,1024),
#     nn.ReLU(),
#     nn.Linear(1024,512),
#     nn.ReLU(),
#     nn.Linear(512,64),
#     nn.ReLU(),
#     nn.Linear(64,4),
#     nn.ReLU(),
#     nn.Linear(4,1),
#     nn.ReLU(),
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super(Model, self).__init__
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,32)
        self.linear4 = nn.Linear(32,16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.relu(x)
        return x

model = Model(8,1).to(DEVICE)
                
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
def train(model, crit_, optim_, x, y):
    optim_.zero_grad()
    hyp_ = model(x)
    loss = crit_(hyp_, y)
    loss.backward()
    optim_.step()
    return loss.item()

    ### fit
epochs = 100
for epoch in range(1,epochs+1):
    loss = train(model, criterion, optimizer, x_tr,y_tr)
    if epochs%11==1:print('epoch_{} Loss : {}'.format(epoch, loss))
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
from sklearn.metrics import r2_score, root_mean_squared_error
y_pred = model(x_ts)
rmse = root_mean_squared_error(y_ts,y_pred.tolist())
r2 = r2_score(y_ts, y_pred.tolist())
print(r2)
