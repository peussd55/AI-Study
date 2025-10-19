### <<44>>

import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 

USE_CUDA = torch.cuda.is_available()    # 상수는 통상적으로 대문자로
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)
# torch : 2.7.1+cu118 사용 device : cuda


x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])
x_pred = np.array([12,13,14])

print(x_train.dtype)          # int32
print(x_train.shape, y_train.shape) # (7,) (7,)

# x = torch.FloatTensor(x).to(DEVICE)
# y = torch.FloatTensor(y).to(DEVICE)

x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)

x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(x_train.size(), y_train.size())   # torch.Size([7, 1]) torch.Size([7, 1])

###### standard scaling ###### 
# dim = axis : 행방향을 따라 이동하며 연산 = 열 단위로 연산. 0번째 차원(축)=shape에서 맨 왼쪽
# keepdim : 차원수 유지여부(True시 계산이 용이)
x_train_mean = torch.mean(x_train, dim=0, keepdim=True)
x_train_std = torch.std(x_train, dim=0, keepdim=True )
x_train = (x_train - x_train_mean) / x_train_std

# test는 sklearn scaler에서 transform만 해야하는 것과 같으므로 train의 mean, std를 쓴다.
x_test = (x_test - x_train_mean) / x_train_std
##############################
print('스케일링 후 :', x_train)

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(1, 32),
    # nn.ReLU(),
    nn.Linear(32, 16),
    # nn.ReLU(),
    nn.Linear(16, 1),
).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

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
    print('epochs: {}, loss: {}'.format(epoch, loss))
    
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
        
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print('최종 loss :', loss2)


### 텐서 예측
x_pred = (torch.Tensor([12,13,14]).unsqueeze(1).to(DEVICE) - x_train_mean) / x_train_std
print(x_pred)
print(type(x_pred))     # <class 'torch.Tensor'>
print(x_pred.size())    # torch.Size([3, 1])
result = model(x_pred)
# detach() : 벡터만 출력하는 item대체 함수. 2차원이상 출력가능
print('[12,13,14]의 예측값 :', result.detach())
# [12,13,14]의 예측값 : tensor([[12.0000],
#         [13.0000],
#         [14.0000]], device='cuda:0')
print('[12,13,14]의 예측값 :', result.detach().cpu().numpy())  # cpu 쓰는 이유 : numpy 연산은 cpu로 해야하기때문에
# [12,13,14]의 예측값 : [[11.999999]
#  [13.      ]
#  [13.999999]]
