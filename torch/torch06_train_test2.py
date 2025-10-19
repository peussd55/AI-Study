### <<44>>

import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 

USE_CUDA = torch.cuda.is_available()    # 상수는 통상적으로 대문자로
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)
# torch : 2.7.1+cu118 사용 device : cuda

x = np.array(range(100))
y = np.array(range(1,101))
x_pred = np.array([101,102])

print(x.dtype)          # int32
print(x.shape, y.shape) # (100,) (100,)
print(x_pred.shape)     # (2,)

# x = torch.FloatTensor(x).to(DEVICE)
# y = torch.FloatTensor(y).to(DEVICE)

x = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(x.size(), y.size())   # torch.Size([100, 1]) torch.Size([100, 1])

# train, test split
train_size = int(len(x) * 0.8)  # 80% 크기 계산
print(train_size)

x_train = x[:train_size]   
y_train = y[:train_size]
x_test = x[train_size:]  
y_test = y[train_size:]

print(x_train.size(), y_train.size(), x_test.size(), y_test.size())     # torch.Size([80, 1]) torch.Size([80, 1]) torch.Size([20, 1]) torch.Size([20, 1])

###### standard scaling #####
# dim = axis : 행방향을 따라 이동하며 연산 = 열 단위로 연산. 0번째 차원(축)=shape에서 맨 왼쪽
# keepdim : 차원수 유지여부(True시 계산이 용이)
x_train_mean = torch.mean(x_train, dim=0, keepdim=True)
x_train_std = torch.std(x_train, dim=0, keepdim=True)
x_train = (x_train - x_train_mean) / x_train_std

# test는 sklearn scaler에서 transform만 해야하는 것과 같으므로 train의 mean, std를 쓴다.
x_test = (x_test - x_train_mean) / x_train_std
##############################
print('스케일링 후 :', x_train)

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(1, 1)
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
x_pred = (torch.Tensor([101,102]).unsqueeze(1).to(DEVICE) - x_train_mean) / x_train_std
print(x_pred)
print(type(x_pred))     # <class 'torch.Tensor'>
print(x_pred.size())    # torch.Size([2, 1])
result = model(x_pred)
# detach() : 벡터만 출력하는 item대체 함수. 2차원이상 출력가능
print('[101,102]의 예측값 :', result.detach())
# [101,102]의 예측값 : tensor([[101.9998],
#         [102.9998]], device='cuda:0')

print('[101,102]의 예측값 :', result.detach().cpu().numpy())  # cpu 쓰는 이유 : numpy 연산은 cpu로 해야하기때문에
# [101,102]의 예측값 : [[101.99979]
#  [102.99978]]
