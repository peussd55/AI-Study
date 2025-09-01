import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
seed = 2497
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda'if USE_CUDA else 'cpu')

###### preparing data
    ### load data
from sklearn.datasets import fetch_california_housing
x, y = fetch_california_housing(return_X_y=True)

    ### train test split
from sklearn.model_selection import train_test_split
x_tr, x_ts, y_tr, y_ts = train_test_split(x,y, train_size=0.8,
                                          shuffle=True, random_state=seed,
                                        #   stratify=y
                                          )
# print(x_tr.shape, y_tr.shape)
    ### scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_tr = sc.fit_transform(x_tr)
x_ts = sc.transform(x_ts)

x_tr = torch.tensor(x_tr, dtype=torch.float32).to(DEVICE)
x_ts = torch.tensor(x_ts, dtype=torch.float32).to(DEVICE)
y_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_ts = torch.tensor(y_ts, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    ### make total tensordataset
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_tr, y_tr)
test_set = TensorDataset(x_ts, y_ts)

    ### make batches set
BATCH_SIZE = 100
train_load = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_load = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
# print(len(train_load), len(test_load))

    ### check batches set from iterator data
# print('='*70)
#     # 1
# for aaa in train_load:
#     x_batch, y_batch = aaa
# print(aaa)
# print('='*70)
#     #2
# bbb = iter(train_load)
# aaa = next(bbb)
# print(aaa)
# print('='*70)
#     # 1-1
# for batch_idx, (x_batch, y_batch) in enumerate(train_load):
#     print(batch_idx)
#     print(x_batch)
#     print(y_batch)
# print('='*70)
    # 2-1   
# first_batch = next(iter(train_load))
# x_batch, y_batch = first_batch
# print('1st x : ' , x_batch.size)
# print('1st y : ', y_batch.size)

###### learn
    ### model
model = nn.Sequential(
    nn.Linear(8,256),
    nn.ReLU(),
    nn.Linear(256,512),
    nn.ReLU(),
    nn.Linear(512,1024),
    nn.ReLU(),
    nn.Linear(1024,512),
    nn.ReLU(),
    nn.Linear(512,64),
    nn.ReLU(),
    nn.Linear(64,16),
    nn.ReLU(),
    nn.Linear(16,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid(),
).to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00012)

def train(model, cirt_, opt_, loader):
    total_loss = 0
    
    for x_batch, y_batch in loader:
        opt_.zero_grad()
        hyp_= model(x_batch)
        loss = cirt_(hyp_,y_batch)
        loss.backward()
        opt_.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

    ### fit
epochs = 20000
for epoch in range(1,epochs+1):
    loss = train(model, criterion, optimizer, train_load)
    if epoch%1000==0:
        print('epoch_{} Loss : {}'.format(epoch, loss))
    else: continue
    
###### eval / pred
    ### eval
from sklearn.metrics import r2_score

def evaluate(model, crit_, loader):
    model.eval()
    total_loss =0 
    y_true_list=[]
    y_pred_list=[]
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(x_batch)
            score = crit_(pred,y_batch)
            
            total_loss += score.item()
            
            y_true_list.extend(y_batch.cpu().numpy())
            y_pred_list.extend(pred.cpu().numpy())
            
    avg_loss = total_loss/len(loader)
    
    y_true_np = np.array(y_true_list).ravel()
    y_pred_np = np.round(np.array(y_pred_list)).astype(int).ravel()
    
    r2 = r2_score(y_true_np, y_pred_np)
    return avg_loss, r2

f_loss = evaluate(model, criterion, test_load)
print('f_loss :', f_loss[0])

    ### pred
from sklearn.metrics import accuracy_score
y_pred = model(x_ts)
r2 = r2_score(y_ts.tolist(), np.round(y_pred.tolist()))
print('acc :', r2)

