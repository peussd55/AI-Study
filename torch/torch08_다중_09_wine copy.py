import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print("torch:", torch.__version__,'사용:', DEVICE)

seed =592

datasets = load_wine()
x=datasets.data
y=datasets.target 

x_train,x_test,y_train,y_test = train_test_split(
    x,y,random_state=seed, shuffle=True,
)

#스케일링
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train=torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test=torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

# y_train=torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
# y_test=torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# y 텐서는 CrossEntropyLoss를 위해 torch.long 타입으로, unsqueeze(1) 없이 변환
y_train=torch.tensor(y_train, dtype=torch.long).to(DEVICE)
y_test=torch.tensor(y_test, dtype=torch.long).to(DEVICE)

print(np.unique(y))
print(len(np.unique(y)))
print(x_train.shape, y_train.shape)

# softmax 값 삭제
# TO(DEVICE)추가
model=nn.Sequential(
    nn.Linear(13,128),
    nn.Linear(128,128),
    nn.ReLU(),
    nn.Linear(128,128),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16, 3)
).to(DEVICE)

#LOSS값은 CrossTentropyloss값으로 수정 
criterion=nn.CrossEntropyLoss()
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

#argmax , DIM=1 추가
y_prd = torch.argmax(y_prd, dim=1).cpu().numpy()
y_test = y_test.cpu().numpy()

acc=accuracy_score( y_prd,y_test)
print('acc:', acc)