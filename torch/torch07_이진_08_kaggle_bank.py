### <<44>>

import numpy as np  
import pandas as pd
import torch
import torch.nn as nn 
import torch.optim as optim 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 device :', DEVICE)

# 1. 데이터
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 인코딩
le_geo = LabelEncoder()
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

# 불필요 컬럼 제거
train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

# train test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=772,
    stratify=y,
)

# 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(type(x_train), type(x_test))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(y_train), type(y_test))  # <class 'pandas.core.series.Series'> <class 'pandas.core.series.Series'>

# 텐서변환 : 안전하게 판다스데이터타입을 nparray로 변환한 후 텐서변환하기 (x_train, x_test는 스케일러할때 numpy변환됐으므로 적용X)
# 판다스데이터(데이터프레임, 시리즈)를 넘파이배열로 바꾸지 않고 바로 텐서변환할 경우 오류나는 경우가 많다.(데이터타입을 정확히 인식못함)
"""
[torch.tensor가 판다스데이터를 텐서변환할때 오류가 안날 조건]
1. 인덱스가 문자형이면 괜찮다 : 문자형일경우 인덱스를 무시하고 값(넘파이배열)만 가지고 연산하기때문에 변환이 된다.
2. 인덱스가 수치형일 경우 : 0을 포함하면서 연속적이어야한다. 둘 중 한 조건이라도 만족을 못하면 ValueError: could not determine the shape of object type 오류가 발생한다. 
 -> train/test분리할 때 shuffle=True 조건으로 분리하면 연속적으로 분리되지 않게 될 확률이 높기때문에 거의 오류가 난다.

=> 결론 : tensor변환할땐 그냥 속편하게 전부 nparray로 바꾸고 하자.
"""
x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1).to(DEVICE)

print("=================================================")
print(x_train.dtype)                # torch.float32
print(x_train.shape, y_train.shape) # torch.Size([132027, 10]) torch.Size([132027, 1])
print(type(x_train))                # <class 'torch.Tensor'>
print("=================================================")

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(10, 64),
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
print(y_predict.size())    # torch.Size([33007, 1])

# 텐서를 넘파이배열로 변환
y_predict = y_predict.detach().cpu().numpy()
y_true = y_test.detach().cpu().numpy()

# 1 또는 0 으로 변환 : astype은 True를 1, False를 0으로 반환
y_predict = (y_predict > 0.5).astype(np.int32)  # 또는 np.round(y_predict)
y_true = y_true.astype(np.int32)

print('accuracy_score :', accuracy_score(y_true, y_predict))
# accuracy_score : 0.8623928257642318
