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
###### preparing data
    ### load data
from sklearn.datasets import fetch_covtype
x, y = fetch_covtype(return_X_y=True)
y = y-1
    ### train test split
from sklearn.model_selection import train_test_split
x_tr, x_ts, y_tr, y_ts = train_test_split(x,y, train_size=0.8,
                                          shuffle=True, random_state=seed,
                                          stratify=y,
                                          )

    ### scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_tr = sc.fit_transform(x_tr)
x_ts = sc.transform(x_ts)
    ### one hot encode
y_tr = pd.get_dummies(y_tr)
print(x_tr.shape, y_tr.shape)
    ### transform to tensor
x_tr = torch.tensor(x_tr, dtype=torch.float32).to(DEVICE)
x_ts = torch.tensor(x_ts, dtype=torch.float32).to(DEVICE)
y_tr = torch.tensor(y_tr.values, dtype=torch.float32).to(DEVICE)
print(x_tr.size(), y_tr.size())
###### learn
    ### model
model = nn.Sequential(
    nn.Linear(54,128),
    nn.ReLU(),
    nn.Linear(128,256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,7),
).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)
def train(model, crit_, optim_, x, y):
    optim_.zero_grad()
    hyp_ = model(x)
    loss = crit_(hyp_, y)
    loss.backward()
    optim_.step()
    return loss.item()

    ### fit
epochs = 1000
for epoch in range(1,epochs+1):
    score = train(model, criterion, optimizer, x_tr, y_tr)
    if epoch%1==0:print('epoch_{} Loss : {}'.format(epoch, score))
    else:continue

###### eval / pred
    ### eval
def evaluate(model, crit_, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss = crit_(y_pred,y)
        return loss.item()
print('f_score : {}'.format(evaluate(model, criterion, x_tr, y_tr)))
    ### pred
from sklearn.metrics import r2_score, accuracy_score
pred = model(x_ts)
acc = accuracy_score(y_ts, np.argmax(pred.tolist(), axis=1))
print('acc :',acc)
