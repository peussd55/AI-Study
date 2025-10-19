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
        
        self.y = self.train_csv['Close'].values.astype(np.float32)
        
        self.timesteps = timesteps
    
    # x시퀀스 길이 : (967, 3) -> (n, 30, 3) : 967 - 30 + 1 (전체 - timestep + 1)
    def __len__(self):                          # 총 사용할수있는 샘플의 갯수 : 967 - 30
        return len(self.x) - self.timesteps     # 0~966인덱스까지 사용가능
    
    # 인덱스 1개에 해당하는 값의 형태를 정의
    def __getitem__(self, idx):
        x = self.x[idx : idx+self.timesteps]     # (1,30,3). 0~29 행
        y = self.y[idx+self.timesteps]           # 30번째 행
        
        # x,y가 nparray이면 tensor 타입으로 변환안해도 DataLoader에서 텐서타입으로 변환하여 반환하나 
        # list같은 파이썬 데이터타입이면 tensor변환을 안해주기때문에 안정성을위해 여기서 명시하는 것이 표준.
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
summary(model, (32, 30, 3))
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# RNN                                      [32, 1]                   --
# ├─RNN: 1-1                               [32, 30, 64]              21,056
# ├─Linear: 1-2                            [32, 32]                  2,080
# ├─ReLU: 1-3                              [32, 32]                  --
# ├─Linear: 1-4                            [32, 1]                   33
# ==========================================================================================
# Total params: 23,169
# Trainable params: 23,169
# Non-trainable params: 0
# Total mult-adds (M): 20.28
# ==========================================================================================
# Input size (MB): 0.01
# Forward/backward pass size (MB): 0.50
# Params size (MB): 0.09
# Estimated Total Size (MB): 0.60
# ==========================================================================================

# 3. 컴파일, 훈련
optim = optim.Adam(model.parameters(), lr=0.0001)

# import tqdm
from tqdm import tqdm

start_time = time.time()
for epoch in range(1, 201):
    iterator = tqdm(train_loader)
    for x, y in iterator:
        # print(x.shape); print(y.shape)  # torch.Size([32, 30, 3]) torch.Size([32])
        optim.zero_grad()
        
        hypothesis = model(x)
        # print(hypothesis.shape)     # torch.Size([32, 1])
        
        # nn.MSELoss()가 y(1차원)를 2차원으로 브로드캐스팅하여 hypothesis와 손실계산
        # warning안뜨게 하려면 y = y.unsqueeze(-1) 적용 : (batch,) -> (batch, 1) 
        loss = nn.MSELoss()(hypothesis, y)
        
        loss.backward()
        optim.step()
        
        iterator.set_description(f'epoch: {epoch}, loss: {round(loss.item(), 5)}')
end_time = time.time()
# epoch: 200, loss: 2399.43091
print('걸린시간 :', end_time-start_time, '초')
# 걸린시간 : 46.38903188705444 초

# save
import os 
print(os.getcwd())
save_path = './_save/torch/'
torch.save(model.state_dict(), save_path + 't25_netflix.pth')
# state_dict : 가중치만 저장
