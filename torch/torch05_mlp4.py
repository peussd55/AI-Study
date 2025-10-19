### <<44>>

import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 

USE_CUDA = torch.cuda.is_available()    # 상수는 통상적으로 대문자로
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)
# torch : 2.7.1+cu118 사용 device : cuda

x = np.array([range(10)])   #(1,10)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [10,9,8,7,6,5,4,3,2,1],
             [9,8,7,6,5,4,3,2,1,0]])
x = x.T
y = y.T
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

print(x.dtype)          # int32
print(x.shape, y.shape) # (10, 1) (10, 3)

# x = torch.FloatTensor(x).to(DEVICE)
# y = torch.FloatTensor(y).to(DEVICE)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

print(x.size(), y.size())   # torch.Size([10, 1]) torch.Size([10, 3])

x_mean = torch.mean(x)
x_std = torch.std(x)

###### standard scaling #####
# dim = axis : 행방향을 따라 이동하며 연산 = 열 단위로 연산. 0번째 차원(축)=shape에서 맨 왼쪽
# keepdim : 차원수 유지여부(True시 계산이 용이)
x_mean = torch.mean(x, dim=0, keepdim=True)     
x_std = torch.std(x, dim=0, keepdim=True)
x = (x - x_mean) / x_std
##############################
print('스케일링 후 :', x)

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(1, 32),
    # nn.ReLU(),
    nn.Linear(32, 16),
    # nn.ReLU(),
    nn.Linear(16, 3),
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
    loss = train(model, criterion, optimizer, x, y)
    print('epochs: {}, loss: {}'.format(epoch, loss))
    
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
        
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss :', loss2)


### 1개 텐서 예측
x_pred1 = (torch.Tensor([[10]]).to(DEVICE) - x_mean) / x_std
print(x_pred1)
print(x_pred1.size())   # torch.Size([1, 1])
result1 = model(x_pred1)
# detach() : 벡터만 출력하는 item대체 함수. 2차원이상 출력가능
print('[[10]]의 예측값 :', result1.detach())
# [10]의 예측값 : tensor([[ 1.1000e+01, -8.9407e-08, -1.0000e+00]], device='cuda:0')
print('[[10]]의 예측값 :', result1.detach().cpu().numpy())  # cpu 쓰는 이유 : numpy 연산은 cpu로 해야하기때문에
# [10]의 예측값 : [[ 1.1000002e+01 -8.9406967e-08 -9.9999917e-01]]
