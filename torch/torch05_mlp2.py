### <<44>>

import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 

USE_CUDA = torch.cuda.is_available()    # 상수는 통상적으로 대문자로
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)
# torch : 2.7.1+cu118 사용 device : cuda

x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]]).T

y = np.array([1,2,3,4,5,6,7,8,9,10])  

print(x.dtype)  # float64
print(x.shape, y.shape) # (10,3) (10,)

# x = torch.FloatTensor(x).to(DEVICE)
# y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
# FloatTensor : 항상 float32 타입으로 변환 -> nn.Linear 연산 dtype과 동일

# 바뀐 권장사항
x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)
# tensor : numpy타입(float64) 그대로 따라감 -> nn.Linear 연산하려면 형변환 필요

print(x.size(), y.size())   # torch.Size([10, 3]) torch.Size([10, 1])

########## standard scaling #########
# dim = axis : 행방향을 따라 이동하며 연산 = 열 단위로 연산. 0번째 차원(축)=shape에서 맨 왼쪽
# keepdim : 차원수 유지여부(True시 계산이 용이)
x_mean = torch.mean(x, dim=0, keepdim=True)     
x_std = torch.std(x, dim=0, keepdim=True)
x = (x - x_mean) / x_std
#####################################
print('스케일링 후 :', x)
# 스케일링 후 : tensor([[-0.9502, -0.9502,  1.7485],
#         [-0.6128, -0.9164,  1.4112],
#         [-0.2755, -0.8827,  1.0739],
#         [ 0.0618, -0.8490,  0.7365],
#         [ 0.3992, -0.8152,  0.3992],
#         [ 0.7365, -0.7815,  0.0618],
#         [ 1.0739, -0.7478, -0.2755],
#         [ 1.4112, -0.7140, -0.6128],
#         [ 1.7485, -0.6803, -0.9502],
#         [ 2.0859, -0.6466, -1.2875]], device='cuda:0', dtype=torch.float64)

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1),
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
# 최종 loss : 0.0

x_pred = (torch.Tensor([[11,2.0,-1],[12,3.0,-2]]).to(DEVICE) - x_mean) / x_std 
print(x_pred)
# tensor([[ 2.4232, -0.6128, -1.6248],
#         [ 2.7605, -0.2755, -1.9622]], device='cuda:0')
result = model(x_pred)

# item() : 요소가 하나인 스칼라값을 반환
print('[11,2.0,-1]의 예측값:', result[0].item())
print('[12,3.0,-2]의 예측값:', result[1].item())
# [11,2.0,-1]의 예측값: 10.999999046325684
# [12,3.0,-2]의 예측값: 12.016620635986328