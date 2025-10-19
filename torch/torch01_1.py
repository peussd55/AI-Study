### <<43>>

import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim  

# 1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# tf와 torch 차이 : tf는 nparray를 변환할 필요없이 바로 연산하지만 torch는 tensor변환을 해야한다.
x = torch.FloatTensor(x)
print(x)            # tensor([1., 2., 3.])
print(x.shape)      # torch.Size([3])
print(x.size())     # torch.Size([3])

# 차원변환 : torch는 tf처럼 자동 브로드캐스팅을 하지 않으므로 않으므로 x,y 행렬변환을 미리 해줘야한다. 또한 벡터를 그대로 연산할 수없고 2차원이상으로 shape변경해야한다.
"""
unsqueeze(i) : i번째 인덱스에 차원(1)추가
ex)
y = torch.randn(2, 3, 4)  # shape: (2, 3, 4)
y0 = y.unsqueeze(0)  # 맨 앞에 추가: (1, 2, 3, 4)
y1 = y.unsqueeze(1)  # 두번째에 추가: (2, 1, 3, 4)
y2 = y.unsqueeze(2)  # 세번째에 추가: (2, 3, 1, 4)
y3 = y.unsqueeze(3)  # 네번째(마지막)에 추가: (2, 3, 4, 1)
"""
x = torch.FloatTensor(x).unsqueeze(1)
print(x)
# tensor([[1.],  
#         [2.],  
#         [3.]])
print(x.shape)      # torch.Size([3, 1])
print(x.size())     # torch.Size([3, 1])
y = torch.FloatTensor(y).unsqueeze(1)
print(y)
# tensor([[1.],
#         [2.],
#         [3.]])
print(y.shape)      # torch.Size([3, 1])
print(y.size())     # torch.Size([3, 1])

# reshape나 view를 사용해서 shape를 직접 지정할 수 있다.
# print((x.reshape(1,1,3,1,1)).shape) # torch.Size([1, 1, 3, 1, 1])
# print((x.view(1,1,3,1,1)).shape)    # torch.Size([1, 1, 3, 1, 1])

"""
[pytorch 문법] 

- reshape
    : unsqueeze()
    : reshape()
    : view()

- shape 확인
    : shape
    : size()

"""

# 2. 모델 구성
# model = Sequential()
# model.add(Dense(1, iniput_dim=1))      # 아웃풋, 인풋
model = nn.Linear(1,1)  # 인풋, 아웃풋    # y = wx + b : 사실은 y = xw +b 이다. x와 w가 행렬인데 wx와 xw는 결과가 다르다.

# 기울기(gradient)와 가중치(weight)는 다르다. -> 여기서 말하는 기울기는 loss를 가중치로 미분한것이다. 
# [SGD] 기준
# 가중치 : w 
# 비용(MSE) : L = (y-y)^2/n  (y = xw+b)
# 기울기 : ∂L/∂w (loss를 weight로 미분)
# 가중치 갱신 : w = w - ln * ∂L/∂w (ln : LearningRate)
 
# 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01) 
optimizer = optim.SGD(model.parameters(), lr=0.01)
# model.parameters() : 학습 가능한 파라미터(즉, 가중치와 편향)”를 모두 iterable(반복가능한 객체)로 반환

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
    
    with torch.no_grad():     # 기울기갱신X. 평가에서 모델을 수행할때 기울기계산을 하지않기 위해 no_grad로 감싼다.
        y_predict = model(x)
        loss2 = criterion(y, y_predict) # loss의 최종값
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss :', loss2)

print(torch.Tensor([[4]]).shape)    # torch.Size([1, 1])
result = model(torch.Tensor([[4]]))
# 또는
# with torch.no_grad():
#     result = model(torch.Tensor([[4]]))
# -> 불필요한 메모리사용제외(기울기계산 제외)
print('4의 예측값 :', result.item())
# 최종 loss : 2.4679715693309845e-07
# 4의 예측값 : 4.000996112823486