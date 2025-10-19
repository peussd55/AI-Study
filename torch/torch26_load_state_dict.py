### <<50>>

import pandas as pd
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import random, time

########################## 랜덤고정 & 쿠다확인 & 데이터종류명시 ##########################
SEED=999
random.seed(SEED)                           # 파이썬 랜덤 고정
np.random.seed(SEED)                        # 넘파이 랜덤 고정
torch.manual_seed(SEED)                     # 토치 랜덤 고정
torch.cuda.manual_seed(SEED)                # 토치 쿠다 랜덤 고정
torch.cuda.manual_seed_all(SEED)            # (멀티 쿠다 사용 시) 모든 쿠다 랜덤 고정

# 토치 cuDNN 랜덤 고정 (transforms 사용시에 필요한 랜덤고정, 비결정론적 알고리즘사용OFF, 성능떨어짐)
# torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)

data_list = ['regression', 'binary_classification', 'multiclass_classification']
data_type = 'regression'
########################## 랜덤고정 & 쿠다확인 & 데이터종류명시 ##########################

# 1. 데이터
# raw데이터로드
path = 'c:/study25/_data/kaggle/netflix/'
train_csv = pd.read_csv(path + 'train.csv')
print(train_csv)    # [967 rows x 6 columns] -> n, 30, 3 으로 바꿀것임.
print(train_csv.info())
print(train_csv.describe())

# 간단한시각화
# import matplotlib.pyplot as plt 
# data = train_csv.iloc[:, 1:4]
# print(data)
# data['종가'] = train_csv['Close']
# print(data)
# hist = data.hist()
# plt.show()

# 커스텀데이터셋구축
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader

class Custom_Dataset(Dataset):
    def __init__(self, df, timesteps):
        self.train_csv = df
        
        # 여기서 float32로 바꿔서 연산했기때문에 __getitem__ 에선 dtype=torch.float32 파라미터안넣어도됨.
        self.x = self.train_csv.iloc[:, 1:4].values.astype(np.float32)
        self.x = (self.x - np.min(self.x, axis=0)) / \
            (np.max(self.x, axis=0) - np.min(self.x, axis=0))   # MinMaxScaler. axis=0 : 행방향을 따라 이동하며 연산 = 열 단위로 연산
        
        self.y = self.train_csv['Close'].values
        
        self.timesteps = timesteps
    
    # x시퀀스 길이 : (967, 3) -> (n, 30, 3) : 967 - 30 + 1 (전체 - timestep + 1)
    def __len__(self):                          # 총 사용할수있는 샘플의 갯수 : 967 - 30
        return len(self.x) - self.timesteps     # 0~966인덱스까지 사용가능
    
    # 인덱스 1개에 해당하는 값의 형태를 정의
    def __getitem__(self, idx):
        x = self.x[idx : idx+self.timesteps]     # (1,30,3). 0~29 행
        y = self.y[idx+self.timesteps]           # 30번째 행
        
        # 여기서 tensor 타입으로 변환안해도 DataLoader에서 텐서타입으로 변환하여 반환하나 안정성을위해 여기서 명시하는 것이 표준.
        x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
        return x, y
    
custom_dataset = Custom_Dataset(df=train_csv, timesteps=30)

train_loader = DataLoader(custom_dataset, batch_size=32, shuffle=False)

for batch_idx, (xb, yb) in enumerate(train_loader):
    print("======== 배치 :", batch_idx, "=========")
    print("xb :", xb.shape)
    print("yb :", yb.shape)
# ...
# ======== 배치 : 28 =========
# xb : torch.Size([32, 30, 3])
# yb : torch.Size([32])
# ======== 배치 : 29 =========
# xb : torch.Size([9, 30, 3])
# yb : torch.Size([9])

# 2. 모델
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.rnn = nn.RNN(input_size=3,
                          hidden_size=64,
                          num_layers=3,
                          batch_first=True,
                          ) # (n, 30, 64)
        self.fc1 = nn.Linear(in_features=30*64, out_features=32)
        # self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.rnn(x)
        
        x = x.reshape(-1, 30*64)
        # x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
model = RNN().to(DEVICE)

# summary 확인
from torchinfo import summary
# summary(model, (32, 30, 3))

# 3. 컴파일, 훈련
"""
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


from tqdm import tqdm
start_time = time.time()
for epoch in range(1, 201):
    iterator = tqdm(train_loader)
    for x, y in iterator:
        optimizer.zero_grad()
        
        hypothesis = model(x)
        loss = criterion(hypothesis, y)
        
        loss.backward()         # 기울기(gradient)값까지만 계산 : ∂L/∂w
        optimizer.step()        # 가중치갱신 : w = w - lr * ∂L/∂w
        
        iterator.set_description(f'epoch : {epoch} loss : {loss.item()}')
end_time = time.time()
print('걸린시간 :', end_time-start_time, '초')
# 걸린시간 : 22.712655067443848 초
"""
# save
save_path = './_save/torch/'
# torch.save(model.state_dict(), save_path + 't25_netflix.pth')

# 4. 평가, 예측
y_predict = []
total_loss = 0
y_true = []


with torch.no_grad():
    model.load_state_dict(torch.load(save_path + 't25_netflix.pth', map_location=DEVICE))
    model.eval()
    for x_test, y_test in train_loader:
        # print(x_test.shape); print(y_test.shape)    # torch.Size([32, 30, 3]) torch.Size([32])

        y_pred = model(x_test.to(DEVICE))
        # print(y_pred.shape)      # torch.Size([32, 1])
        y_predict.append(y_pred.cpu().numpy())
        
        # print(y_test.shape)       # torch.Size([32])
        y_true.append(y_test.cpu().numpy())
        
        y_test = y_test.unsqueeze(-1)  # (batch,) -> (batch, 1)
        loss = nn.MSELoss()(y_pred, y_test.to(DEVICE))
        print("loss :", loss)
        total_loss += loss / len(train_loader)
        # total_loss = total_loss + (loss / len(train_loader))
        # 12.01/30 + 36.83/30 + .. + 50.0/30
        
print('평균 loss :', total_loss, "/", total_loss.item())    # tensor(10602.3916, device='cuda:0') / 10602.3916015625

from sklearn.metrics import r2_score

print(type(y_predict))  # <class 'list'> : [(32, 1), (32, 1), ..., (9,1)] -> shape가 동일하지 않은게(맨 마지막)이 있어서 바로 np.array적용불가
print(type(y_true))     # <class 'list'> : [(32, 1), (32, 1), ..., (9,1)] -> shape가 동일하지 않은게(맨 마지막)이 있어서 바로 np.array적용불가
# [
#   array([[...], ...], shape=(32,1)),
#   array([[...], ...], shape=(32,1)),
#   ...
#   array([[...], ...], shape=(9,1))
# ]

y_predict = np.concatenate(y_predict, axis=0)   # (전체 샘플 수, 1)
print(y_predict.shape)                          # (937, 1)

y_true = np.concatenate(y_true, axis=0)         # (전체 샘플 수, 1)
print(y_true.shape)                             # (937,)

r2 = r2_score(y_true, y_predict)
print('R2 :', r2)
print('total_loss :', total_loss.item())
# R2 : 0.9277747315665769
# total_loss : 817.9605712890625
# -> rnn에서 flatten안한 모델 load해서 추론하면 r2 마이너스값이 나오고 loss 1만넘음

"""
[고찰]
1. 시드를 고정해도 학습코드랑 추론코드에서 스코어와 loss가 완전히 같게 나오지 않을 수있다.

2. 데이터가 적을때는 RNN에 timesteps를 넣어서 훈련시키면 성능이 극도로 안좋아진다. (Underfitting)
-> 가중치를 저장하고 불러와서 추론하면 r2 스코어가 마이너스가 나올정도로 추론을 제대로 못한다.
--> (대안) Linear로만 구성하거나 RNN의 구조를 timesteps*in_features 로 flatten(reshape)해야 추론 성능이 어느정도 나온다.
"""

"""
[집계함수(min, max), 합치는함수(concatenate)에서 axis(dim) 파라미터를 다루는 연산방식 차이]

1. 집계(axis=0):
    "모든 행(row)을 따라 계산해서, 열(컬럼)별 결과만 남김"
2. 합침(axis=0):
    "모든 행(row)으로 데이터를 아래로 쭉 쌓아 전체 샘플 수(행)를 늘림"
"""