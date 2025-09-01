import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim


USE_CUDA=torch.cuda.is_available()
DEVICE=torch.device('cuda'if USE_CUDA else 'cpu')
print("torch:", torch.__version__,'사용 device:', DEVICE )

#1 데이터 
x = np.array([1,2,3])
y = np.array([1,2,3])

#토치로 변환 
x =torch.FloatTensor(x)

# print(x)
#tensor([1., 2., 3.])
# print(x.shape)
# print(x.size())
# torch.Size([3]).
x=torch.FloatTensor(x).unsqueeze(1).to(DEVICE)

# unsqueeze() 함수는 텐서에 새로운 차원(dimension)을 추가하는 데 사용됩니다. 
# 특정 위치에 크기가 1인 차원을 삽입하여 텐서의 모양(shape)을 변경할 수 있습니다. 
# 이는 특히 모델 입력 형태를 맞추거나 배치 차원을 추가할 때 유용합니다.

# print(x)
# print(x.shape)
# print(x.size())

# tensor([[1.],
#         [2.],
#         [3.]])
# torch.Size([3, 1])
# torch.Size([3, 1])

y=torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
# print(y.shape)
# torch.Size([3, 1])

#모델 구성
model=nn.Linear(1,1).to(DEVICE) # 앞에가 input, 뒤에가 아웃풋 # y= xw + b

#컴파일, 훈련

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.038) #경사하감법

def train(model, criterion, optimizer, x, y):
    # model.train() #[훈련모드], 드랍아웃, 배치노말 사용가능
    optimizer.zero_grad() # 기울기 초기화
                          # 각 배치마다 기울기를 초기화(0으로)하여, 기울기 누적에 의한 문제 해결
    hypothesis=model(x) # y=wx+b , 예측값
    loss=criterion(hypothesis, y)  # loss = mse() -> 시그마(y -hypothesis ^/2n)
    ########여기까지가 순전파, 지금부터 역전파

    loss.backward() # 기울기(gradient)까지만 계산,
    optimizer.step() #가중치 갱신
    
    return loss.item()
    
epochs=100
for epoch in range(1,100):
    loss=train(model, criterion,optimizer,x,y)
    print('epochs:{}, loss:{}'. format(epoch, loss))
    
    
print("======================================평가==========================================")
#4 평가, 예측

def evaluate(model, criterion, x,y):
    model.eval() # [훈련모드] 드랍아웃, 배치노말아이즈 쓰지 않는다.
    with torch.no_grad():  # 기울기 갱신 하지 않겠다. 
        y_predict = model(x)
        loss2 = criterion(y, y_predict) # loss의 최종값
    return loss2.item()

loss2 = evaluate(model,criterion,x,y)
print("최종loss:",loss2)

results=model(torch.Tensor([[4]]).to(DEVICE))
print("4의 예측값", results.item())