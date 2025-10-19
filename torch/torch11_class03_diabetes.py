### <<45>>

import numpy as np  
import random
import torch
import torch.nn as nn 
import pandas as pd
import torch.optim as optim 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

########################## 랜덤고정 & 쿠다확인 & 데이터종류명시 ##########################
SEED=999
random.seed(SEED)               # 파이썬 랜덤 고정
np.random.seed(SEED)            # 넘파이 랜덤 고정
torch.manual_seed(SEED)         # 토치 랜덤 고정
torch.cuda.manual_seed(SEED)    # 토치 쿠다 랜덤 고정

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)

data_list = ['regression', 'binary_classification', 'multiclass_classification']
data_type = 'regression'
########################## 랜덤고정 & 쿠다확인 & 데이터종류명시 ##########################

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (442, 10) (442,)

# train test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=SEED,
    stratify=y if data_type != 'regression' else None,
)

# 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# nparray로 사전변환
x_train, x_test, y_train, y_test = [
    np.array(v) if not isinstance(v, np.ndarray) else v
    for v in [x_train, x_test, y_train, y_test]
]

# 텐서변환
x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
if data_type == 'multiclass_classification':
    # 다중 분류: 1D, long 타입으로 변환 (unsqueeze 없음)
    y_train, y_test = [
        torch.tensor(v, dtype=torch.long).to(DEVICE) for v in [y_train, y_test]
    ]
else:
    # 회귀 또는 이진 분류: 2D (unsqueeze 적용), float 타입으로 변환
    y_train, y_test = [
        torch.tensor(v, dtype=torch.float32).unsqueeze(1).to(DEVICE) for v in [y_train, y_test]
    ]
print("==================== 최종 shape =====================")
print(x_train.shape, y_train.shape)
print(type(x_train), type(y_train))
# torch.Size([309, 10]) torch.Size([309, 1])
# <class 'torch.Tensor'> <class 'torch.Tensor'>

# 하이퍼파라미터 설정
input_dim = x_train.shape[1]
output_dim = len(np.unique(y)) if data_type == 'multiclass_classification' else y_train.shape[1]
lr=0.01
epochs = 300
criterion_map = {
    'regression': nn.MSELoss,
    'binary_classification': nn.BCELoss,
    'multiclass_classification': nn.CrossEntropyLoss
}
criterion = criterion_map[data_type]()
print("==================== (하이퍼)파라미터 =====================")
print("input_dim :", input_dim)
print("output_dim :", output_dim)
print("learning_rate :", lr)
print("epochs :", epochs)
print("criterion :", 'nn.' + str(criterion))
# input_dim : 10
# output_dim : 1
# learning_rate : 0.01
# epochs : 300
# criterion : nn.MSELoss()

# exit()
# 2. 모델구성
class Model(nn.Module):                             
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # 훈련레이어
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        
        # 활성화함수
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        
        # 드롭아웃
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        # 학습, 배치정규화 레이어 : 중첩사용X (가중치 편향문제)
        # 활성화함수, 드롭아웃 레이어 : 중첩사용O
        """
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        
        x = self.linear2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        
        x = self.linear3(x)
        x = self.relu(x)
        # x = self.dropout(x)
        
        x = self.linear4(x)
        # x = self.sigmoid(x)
        
        return x

model = Model(input_dim, output_dim).to(DEVICE)
print(model)

# 3. 컴파일, 훈련
criterion = criterion
optimizer = optim.Adam(model.parameters(), lr=lr)

def train(model, criterion, optimizer, x, y):
    # model.train()
    optimizer.zero_grad()
    
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = epochs
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epochs: {}, loss: {}'.format(epoch, loss))
print("======================================")

# 4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y_predict, y)
        
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print('최종 loss :', last_loss)     # 2609.389892578125

# target 생성
y_predict = model(x_test)
print(y_predict.size())             # torch.Size([133, 1])

# 텐서를 넘파이배열로 변환(output, target)
y_predict = y_predict.detach().cpu().numpy()
y_true = y_test.detach().cpu().numpy()

# target 변환
if data_type == 'binary_classification':
    # 1 또는 0 으로 변환 : astype은 True를 1, False를 0으로 반환
    y_predict = (y_predict > 0.5).astype(np.int32)  # 또는 np.round(y_predict)
elif data_type == 'multiclass_classification':
    y_predict = np.argmax(y_predict, axis=1)
else:
    y_predict = y_predict
    
# 예측스코어 확인
if data_type == 'regression':
    print('r2_score :', r2_score(y_true, y_predict))
else:
    print('acc :', accuracy_score(y_true, y_predict))

""" 
[최종스코어]
r2_score : 0.5100261569023132

"""