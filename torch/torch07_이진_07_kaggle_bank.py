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

path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder() # 인스턴스화
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])   # fit 함수 + transform 함친 합친 함수 : 변환해서 적용
# # 아래 2줄이랑 같다.
# le_geo.fit(train_csv['Geography'])                                    # 'Geography' 컬럼을 기준으로 인코딩한다.
# train_csv['Geography'] = le_geo.transform(train_csv['Geography'])     # 적용하고 train_csv['컬럼']에 입력함.
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])

# 테스트 데이터도 수치화해야한다. 위에서 인스턴스가 이미 fit해놨기때문에 transform만 적용한다.
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])
train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)  
y = train_csv['Exited']

x_train, x_test, y_train , y_test=train_test_split(
    x,y, random_state=seed, train_size=0.7, shuffle=True
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test =  scaler.transform(x_test)


print(x_train.shape) #(115523, 10)

x_train=torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test=torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train=torch.tensor(y_train.values , dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test=torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape)
# torch.Size([398, 30]) (398,)

model=nn.Sequential(    
    nn.Linear(10,64),
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