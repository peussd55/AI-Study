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

seed =592
# 1. 데이터
path = './_data/dacon/따릉이/'          

train_csv =  pd.read_csv(path + 'train.csv', index_col=0)     
test_csv = pd.read_csv(path + 'test.csv', index_col=0)  
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)  
y = train_csv['count']

x_train, x_test, y_train, y_test=train_test_split(
    x,
    y,
    random_state=seed, 
    test_size=0.8
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train=torch.tensor(x_train.values, dtype=torch.float32).to(DEVICE)
x_test=torch.tensor(x_test.values, dtype=torch.float32).to(DEVICE)

y_train=torch.tensor(y_train.values, dtype=torch.float32).to(DEVICE)
y_test=torch.tensor(y_test.values, dtype=torch.float32).to(DEVICE)


# model=nn.Sequential(
#     nn.Linear(8,128),
#     nn.ReLU(),
#     nn.Linear(128,128),
#     nn.ReLU(),
#     nn.Linear(128,64),
#     nn.ReLU(),
#     nn.Linear(64,64),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.Linear(16,1),
    
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # super(Model, self).__init__() #nn.Module에 있는 model과 self 다 사용
        ### 모델에 대한 정의 부분###
        self.liner1=nn.Linear(input_dim,64)
        self.liner2=nn.Linear(64,32)
        self.liner3=nn.Linear(32,32)
        self.liner4=nn.Linear(32,16)
        self.liner5=nn.Linear(16,output_dim)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.2)
        
    def forward(self, x): # 정의 구현 
        x=self.liner1(x)
        x=self.relu(x)        
        x=self.liner2(x)  
        x=self.relu(x)        
        x=self.liner3(x) 
        x=self.relu(x)        
        x=self.liner4(x)
        x=self.relu(x)        
        x=self.liner5(x)
        
        return x
model=Model(9,1).to(DEVICE)
        
criterion= nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=0.02)

def train(model, criterion, optimizer, x,y):
    #model.train(),
    optimizer.zero_grad()
    hypothesis=model(x)
    loss=criterion(hypothesis,y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs=100
for epoch in range(1,epochs+1):
    loss=train(model, criterion,optimizer,x_train, y_train)
    print("epoch:{}, loss:{}".format(epoch, loss))
    
    
def evalute(model, criterion, x,y,):   
    model.eval()
    with torch.no_grad():
        y_prd=model(x)
        loss2=criterion(y_prd,y)
        
        return loss2.item()
    
results = evalute(model,criterion,x_test,y_test)
print("loss:", results)

y_prd=model(x_test)

y_prd=y_prd.detach().cpu().numpy()
y_test=y_test.detach().cpu().numpy()

print(y_prd, y_test)

# R2 Score 계산
r2 = r2_score(y_test, y_prd)
print(f'R2 Score (Test Set): {r2:.4f}')

# MSE (sklearn 버전) 계산 - PyTorch MSE Loss 값과 동일해야 함
mse_sklearn = mean_squared_error(y_test, y_prd)
print(f'MSE (Sklearn): {mse_sklearn:.4f}')