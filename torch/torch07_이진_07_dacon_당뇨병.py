import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print("torch:", torch.__version__,'사용:', DEVICE)

seed=222

path = './_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

x = train_csv.drop(['Outcome'], axis=1) # (652, 9)
y = train_csv['Outcome']      

x_train, x_test, y_trian, y_test=train_test_split(
    x,y, random_state=seed, train_size=0.7, shuffle=True
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test =  scaler.transform(x_test)

x_train=torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test=torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train=torch.tensor(y_trian, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test=torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_trian.shape)
# torch.Size([456, 8]) (456,)

model=nn.Sequential(    
    nn.Linear(8,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.SiLU(),
    nn.Linear(16,1),
    nn.Sigmoid()
).to(DEVICE)

#컴파일, 훈련
criterion=nn.BCELoss()
optimizer =optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer,x, y):
    #model.train()
    optimizer.zero_grad()
    hypothesis=model(x)
    loss=criterion(hypothesis,y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()
epochs=200
for epoch in range(1,epochs+1):
    loss=train(model, criterion, optimizer, x_train, y_train)
    print("epochs:{}, loss:{}".format(epoch, loss))
    
#평가, 예측

def evaluate(model, criterion,x,y):
    model.eval()
    with torch.no_grad():
        y_prd=model(x)
        loss2=criterion(y_prd,y)
    
    return loss2.item()
results=evaluate(model, criterion,x_test, y_test)
print("최종loss:", results)

y_prd=model(x_test)

y_prd = np.round(y_prd.detach().cpu().numpy())
y_test = y_test.cpu().numpy()

acc=accuracy_score( y_prd,y_test)
print('acc:', acc)