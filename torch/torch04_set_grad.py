### <<43>>

import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim  

USE_CUDA  = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)
# 파이토치에서는 cuda를 사용하려면 데이터(x, y)와 모델에 전부 cuda를 사용한다고 명시해줘야한다. (to(DEVICE))

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1 ,1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]
             ).transpose()
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x는 이미 2차원이상 이므로 unsqueeze 불필요
x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

print(x.size(), y.size())   # torch.Size([10, 2]) torch.Size([10, 1])

########## standard scaling ##########
# dim = axis : 행방향을 따라 이동하며 연산 = 열 단위로 연산. 0번째 차원(축)=shape에서 맨 왼쪽
# keepdim : 차원수 유지여부(True시 계산이 용이)
x_mean = torch.mean(x, dim=0, keepdim=True)     
x_std = torch.std(x, dim=0, keepdim=True)
x = (x - x_mean) / x_std
######################################
print('스케일링 후 :', x)
# 스케일링 후 : tensor([[-1.],
#         [ 0.],
#         [ 1.]], device='cuda:0')

# 2. 모델 구성
# model.add(Dense(1, iniput_dim=1))      # 아웃풋, 인풋
# model = nn.Linear(1,5).to(DEVICE)  # 인풋, 아웃풋

model = nn.Sequential(
    nn.Linear(2, 5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1),
).to(DEVICE)
 
# 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01) 
optimizer = optim.SGD(model.parameters(), lr=0.01)  # model의 모든 파라미터에 SGD를 적용

print(model.parameters) # 모든 파라미터
# <bound method Module.parameters of Sequential(
#   (0): Linear(in_features=2, out_features=5, bias=True)
#   (1): Linear(in_features=5, out_features=4, bias=True)
#   (2): Linear(in_features=4, out_features=3, bias=True)
#   (3): Linear(in_features=3, out_features=2, bias=True)
#   (4): Linear(in_features=2, out_features=1, bias=True)
# )

def train(model, criterion, optimizer, x, y):
    # model.train()                     # 디폴트. [훈련모드] : 드랍아웃, 배치노말 적용
    optimizer.zero_grad()               # 기울기 초기화
                                        # 각 배치마다 기울기를 초기화(0으로)하여, 기울기 누적에 의한 문제 해결
    hypothesis = model(x)               # y = xw + b.   # hypothsis : predict와 유사
    loss = criterion(hypothesis, y)     # loss = mse() = 시그마(y-hypothesis)^2/n

    loss.backward()                     # 기울기(gradient)값까지만 계산 : ∂L/∂w
    optimizer.step()                    # 가중치 갱신 : w = w - ln * ∂L/∂w
    
    return loss.item()
    # 위의 한 스텝이 1epoch

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epochs: {}, loss: {}'.format(epoch, loss))
    
print("====================================================")
# 4. 평가, 예측
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):
    model.eval()        # 필수적용. [평가모드] : 드랍아웃, 배치노말을 쓰지 않겠다.(평가에선 절대 드랍아웃, 배치노말 사용X)
    
    # with torch.no_grad():     # 기울기갱신X
    #     y_predict = model(x)
    #     loss2 = criterion(y, y_predict) # loss의 최종값
    
    # 위 구문을 풀어쓰면 다음과같다.
    torch.set_grad_enabled(False)
    y_predict = model(x)
    loss2 = criterion(y, y_predict) # loss의 최종값
    torch.set_grad_enabled(True)
    
    """
    y_predict = model(x)
    loss2 = criterion(y, y_predict) # loss의 최종값
    만 써도 되지만 파이토치의 디폴트 옵션인 자동미분계산을 끄기위해서 사용한다.
    -> 안끄면 evaluate때 가중치 갱신이 한번 더 되어서 뒤의 계산에 영향을 미친다.
    """
    
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss :', loss2)


x_pred = (torch.Tensor([[10, 1.3]]).to(DEVICE) - x_mean) / x_std
print(x_pred)   # tensor([[ 2.2030, -0.7076]], device='cuda:0')
result = model(x_pred)  

print('[10, 1.3]의 예측값:', result.item())
# [10, 1.3]의 예측값:  9.999716758728027