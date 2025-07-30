### <<48>>

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR100
import random, time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

########################## 랜덤고정 & 쿠다확인 & 데이터종류명시 ##########################
SEED=999
random.seed(SEED)                           # 파이썬 랜덤 고정
np.random.seed(SEED)                        # 넘파이 랜덤 고정
torch.manual_seed(SEED)                     # 토치 랜덤 고정
torch.cuda.manual_seed(SEED)                # 토치 쿠다 랜덤 고정
torch.cuda.manual_seed_all(SEED)            # (멀티 쿠다 사용 시) 모든 쿠다 랜덤 고정

# 토치 cuDNN 랜덤 고정 (transforms 사용시에 필요한 랜덤고정, 비결정론적 알고리즘사용OFF, 성능떨어짐)
# torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)

data_list = ['regression', 'binary_classification', 'multiclass_classification']
data_type = 'multiclass_classification'
########################## 랜덤고정 & 쿠다확인 & 데이터종류명시 ##########################

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.array([
                [1,2,3],    
                [2,3,4],
                [3,4,5],
                [4,5,6],
                [5,6,7],
                [6,7,8],
                [7,8,9],
            ])  # 7x3. timesteps은 3
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)     # (7, 3) (7,)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)              # (7, 3, 1)

# x = torch.FloatTensor(x).to(DEVICE)
x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)
print(x.shape, y.size())    # torch.Size([7, 3, 1]) torch.Size([7, 1])

from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

aaa = iter(train_loader)
bbb = next(aaa)
print(bbb)
# [tensor([[[3.],
#          [4.],
#          [5.]],

#         [[7.],
#          [8.],
#          [9.]]], device='cuda:0'), tensor([[ 6.],
#         [10.]], device='cuda:0')]

# 2. 모델
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_layer1 = nn.RNN(
            input_size=1,       # feature 갯수. 텐서플로우에서는 input_dim
            hidden_size=32,     # output_node의 갯수. 텐서플로우에서는 unit
            # num_layers=1,     # 디폴트=1. RNN 은닉층의 레이어의 갯수
            batch_first=True,   # rnn input 파라미터 기본 순서 : TimeSteps, Feature, Batch 을 -> Batch, Timesteps, Feature로 변경. 무조건 넣어주는옵션으로 생각.
                                # (3, 1, N) -> (N, 3, 1)
        )   # (N, 3, 32)
        # 또는 self.rnn_layer1 = nn.RNN(1, 32, batch_first=True)
        
        self.fc1 = nn.Linear(3*32, 16)   # timesteps * feature : 이렇게해도 오류는 나지 않으나 시간의 순서가 무시되어 학습된다.
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
   
    def forward(self, x):
        x, _ = self.rnn_layer1(x)   # ouput이 원래는 2개이기때문에 2개로 받아야함(y로 가는거, hidden으로 가는거. 그러나 hidden으로 가는건 거의 사용할 일없으므로 _ 로 받는다.)
        x = self.relu(x)
       
        x = x.reshape(-1, 3*32)     
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
       
        return x
   
model = RNN().to(DEVICE)

# summary 확인
from torchinfo import summary
summary(model, (2, 3, 1))
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# RNN                                      [2, 1]                    --
# ├─RNN: 1-1                               [2, 3, 32]                1,120
# ├─ReLU: 1-2                              [2, 3, 32]                --
# ├─Linear: 1-3                            [2, 16]                   1,552
# ├─Linear: 1-4                            [2, 8]                    136
# ├─ReLU: 1-5                              [2, 8]                    --
# ├─Linear: 1-6                            [2, 1]                    9
# ==========================================================================================
# Total params: 2,817
# Trainable params: 2,817
# Non-trainable params: 0
# Total mult-adds (M): 0.01
# ==========================================================================================
# Input size (MB): 0.00
# Forward/backward pass size (MB): 0.00
# Params size (MB): 0.01
# Estimated Total Size (MB): 0.01
# ==========================================================================================

# exit()
# 3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(model, criterion, optimizer, loader):
    model.train()   # Dropout, BatchNorm 등의 레이어를 활성화. 디폴트이지만 2epoch부터 eval()로 남아있는 것 방지 위해 명시해줘야함.
    epoch_loss = 0
        
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch, y_batch

        optimizer.zero_grad()   # 기울기 0으로 초기화
        
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()         # 기울기(gradient)값까지만 계산 : ∂L/∂w
        optimizer.step()        # 가중치갱신 : w = w - lr * ∂L/∂w
        
        epoch_loss += loss.item()
                
    return epoch_loss/len(loader)

epochs = 2000
start_time = time.time()
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epochs: {}, loss: {}'.format(epoch, loss))
end_time = time.time()
print('걸린시간 :', end_time-start_time, '초')
# 걸린시간 : 17.873060703277588 초

def evaluate(model, criterion, loader):
    model.eval()    # Dropout, BatchNorm 등의 레이어를 비활성화
    epoch_loss = 0
    
    with torch.no_grad():   # 기울기계산기록X (메모리폭증, 불필요한 연산제거)
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch, y_batch
            
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            
            epoch_loss += loss.item()
        
    return epoch_loss/len(loader)

# 4. 평가, 예측
# 테스트데이터없어서 훈련데이터로 대체
loss2 = evaluate(model, criterion, train_loader)
print('최종 loss :', loss2)
# 최종 loss : 8.000835751431623e-05

x_pred = torch.Tensor([[[8],[9],[10]]]).to(DEVICE)  # 모델의 timestep이 3이기때문에 3개 연속의 숫자를 넣어야함.
print(x_pred)
print(x_pred.shape)
# torch.Size([1, 3, 1])
result = model(x_pred)

# item() : 요소가 하나인 스칼라값을 반환
print('[8, 9, 10]의 예측값:', result.item())
# [8, 9, 10]의 예측값: 10.701749801635742
