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
D_path = 'c:/study25/_data/dacon/당뇨병/'
train_csv = pd.read_csv(D_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(D_path + 'test.csv', index_col=0)
x = train_csv.drop(columns='Outcome')
y = train_csv['Outcome']

    ### train test split
from sklearn.model_selection import train_test_split
x_tr, x_ts, y_tr, y_ts = train_test_split(x,y, train_size=0.8,
                                          shuffle=True, random_state=seed,
                                          stratify=y)

    ### scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_tr = sc.fit_transform(x_tr)
x_ts = sc.transform(x_ts)

x_tr = torch.tensor(x_tr, dtype=torch.float32).to(DEVICE)
x_ts = torch.tensor(x_ts, dtype=torch.float32).to(DEVICE)
y_tr = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_ts = torch.tensor(y_ts.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

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

def train(model, cirt_, opt_, x, y):
    opt_.zero_grad()
    hyp_= model(x)
    loss = cirt_(hyp_,y)
    loss.backward()
    opt_.step()
    return loss.item()
    ### fit
epochs = 20000
for epoch in range(1,epochs+1):
    loss = train(model, criterion, optimizer, x_tr, y_tr)
    if epoch%1000==0:
        print('epoch_{} Loss : {}'.format(epoch, loss))
    else: continue
    
###### eval / pred
    ### eval
def evaluate(model, crit_, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        score = crit_(y,pred)
        return score.item()
f_loss = evaluate(model, criterion, x_tr, y_tr)
print('f_loss :', f_loss)

    ### pred
from sklearn.metrics import accuracy_score
y_pred = model(x_ts)
acc = accuracy_score(y_ts.tolist(), np.round(y_pred.tolist()))
print('acc :', acc)
        

