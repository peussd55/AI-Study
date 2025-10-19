### <<44>>

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
print('torch : ', torch.__version__, '사용 device :', DEVICE)

# 1. 데이터
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 결측치 처리
x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan).fillna(x.median())  # 0값을 NaN으로 변환 후 중앙값 대체
y = train_csv['Outcome']

# train test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=777,
    stratify=y,
)

# 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(type(x_train), type(x_test))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(y_train), type(y_test))  # <class 'pandas.core.series.Series'> <class 'pandas.core.series.Series'>
print(y_train)

# 텐서변환 : 안전하게 판다스데이터타입을 nparray로 변환한 후 텐서변환하기 (x_train, x_test는 스케일러할때 numpy변환됐으므로 적용X)
x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1).to(DEVICE)

print("=================================================")
print(x_train.dtype)                # torch.float32
print(x_train.shape, y_train.shape) # torch.Size([456, 8]) torch.Size([456, 1])
print(type(x_train))                # <class 'torch.Tensor'>
print("=================================================")

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.SiLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    # model.train()
    optimizer.zero_grad()
    
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epochs: {}, loss: {}'.format(epoch, loss))   # verbose
print("==============================================")

# 4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y_predict, y)
        
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print('최종 loss :', last_loss)

y_predict = model(x_test)
print(type(y_predict))     # <class 'torch.Tensor'>
print(y_predict.size())    # torch.Size([169, 1])

# 텐서를 넘파이배열로 변환
y_predict = y_predict.detach().cpu().numpy()
y_true = y_test.detach().cpu().numpy()

# 1 또는 0 으로 변환 : astype은 True를 1, False를 0으로 반환
y_predict = (y_predict > 0.5).astype(np.int32)  # 또는 np.round(y_predict)
y_true = y_true.astype(np.int32)

print('accuracy_score :', accuracy_score(y_true, y_predict))
# accuracy_score : 0.7040816326530612
