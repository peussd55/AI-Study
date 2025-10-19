### <<46>>

import numpy as np  
import random, time
import torch
import torch.nn as nn 
import pandas as pd
import torch.optim as optim 
from sklearn.datasets import fetch_california_housing
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
datasets = fetch_california_housing()
x = datasets.data 
y = datasets.target
print(x.shape, y.shape)

# train test 분리
train_size = 0.7
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=train_size, shuffle=True, random_state=SEED,
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
print(x_test.shape, y_test.shape)
print(type(x_train), type(y_train))
# torch.Size([14447, 8]) torch.Size([14447, 1])
# torch.Size([6193, 8]) torch.Size([6193, 1])
# <class 'torch.Tensor'> <class 'torch.Tensor'>

# 토치데이터셋 생성
from torch.utils.data import TensorDataset  # x,y 합치기
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

# 하이퍼파라미터 설정
input_dim = x_train.shape[1]
output_dim = len(np.unique(y)) if data_type == 'multiclass_classification' else y_train.shape[1]
lr=0.01
epochs = 500
criterion_map = {
    'regression': nn.MSELoss,
    'binary_classification': nn.BCELoss,
    'multiclass_classification': nn.CrossEntropyLoss
}
criterion = criterion_map[data_type]()
batch_size = 10000

# 토치데이터로더 생성
from torch.utils.data import DataLoader                                     # batch 정의
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)   # batch_size : 미니배치
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
# test_loader에서 shuffle=False : 재현성보장 -> 학습과정 추적용이
# train_loader에서 shuffle=True : 매 epoch마다 학습데이터를 섞어서 배치순서의존도를 없애고 과적합 피하기위함

# (하이퍼)파라미터 & 데이터 정보 출력
print("==================== (하이퍼)파라미터 & 데이터 정보 출력 =====================")
print("[파라미터] input_dim :", input_dim)
print("[파라미터] output_dim :", output_dim)
print("----------------")
print("[하이퍼 파라미터] learning_rate :", lr)
print("[하이퍼 파라미터] epochs :", epochs)
print("[하이퍼 파라미터] criterion :", 'nn.' + str(criterion))
print("[하이퍼 파라미터] 배치사이즈 :", batch_size)
print("----------------")
print("원시데이터크기 :", len(x))
print("train size :", train_size)
print("train 갯수 :", len(x_train))
print("배치갯수 :", len(train_loader))
# [파라미터] input_dim : 8
# [파라미터] output_dim : 1
# ----------------
# [하이퍼 파라미터] learning_rate : 0.01
# [하이퍼 파라미터] epochs : 500
# [하이퍼 파라미터] criterion : nn.MSELoss()
# [하이퍼 파라미터] 배치사이즈 : 10000
# ----------------
# 원시데이터크기 : 20640
# train size : 0.7
# train 갯수 : 14447
# 배치갯수 : 452

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

start_time = time.time()
def train(model, criterion, optimizer, loader):
    # model.train()
    
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)   # total_loss없이 loss만 쓰면 문제점: for문안에있어서 마지막 batch의 loss로 덮어씌워짐.
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)             # 모든 batch loss의 평균을 반환함

epochs = epochs
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epochs: {}, loss: {}'.format(epoch, loss))
print("======================================")
end_time = time.time()
print("걸린시간 :", end_time - start_time, "초") 

# 4.1 평가
def evaluate(model, criterion, loader):
    model.eval()
    
    total_loss = 0
    with torch.no_grad():                       # no_grad는 for문 바깥에 사용하는걸 추천
        for x_batch, y_batch in loader:
            y_predict = model(x_batch)
            loss2 = criterion(y_predict, y_batch)
            total_loss += loss2.item()
            
    return total_loss / len(loader)             # 모든 batch loss의 평균을 반환함

last_loss = evaluate(model, criterion, test_loader)
print('최종 loss :', last_loss)

# 4.2 예측
# target 생성
model.eval()            # eval모드로 변환
all_predictions = []    # 예측결과를 담을 리스트생성
with torch.no_grad():   # 배치단위로 예측
    for x_batch, y_batch_dummy in test_loader:
        y_predict_batch = model(x_batch)
        all_predictions.append(y_predict_batch.cpu())   # [텐서1, 텐서2, 텐서3, ...]
# 텐서를 넘파이배열로 변환(output, target)
y_predict = torch.cat(all_predictions, dim=0).numpy()   # torch.cat : 행단위로 텐서1 + 텐서2 + 텐서3 + ... 합한 후 numpy로 변환
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
걸린시간 : 75.9252679347992 초
최종 loss : 0.2960793077945709
r2_score : 0.7842627167701721
"""