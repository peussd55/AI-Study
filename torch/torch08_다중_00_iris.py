### <<45>>

import numpy as np  
import torch
import torch.nn as nn 
import torch.optim as optim 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 device :', DEVICE)

# 1. 데이터
datasets = load_iris()
x = datasets.data 
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)
print(np.unique(y, return_counts=True))     # (array([0, 1, 2]), array([50, 50, 50], dtype=int64))

# train test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=777,
    stratify=y,
)

# 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 텐서변환
x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

# y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
# y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# CrossEntropyLoss() 손실함수는 y 1차원 입력이 필요. 또한 CrossEntropyLoss() 손실함수는 long연산을 해야하기때문에 Target Label은 long(=int64)으로 변환필요
# unsqueeze(1)로 차원추가 안하는 이유 : y를 원핫인코딩해야한다면 모델 출력레이어 output과 컬럼수 맞춰기위해 해야하지만
# CrossEntropyLoss함수는 Sparse_Categorical_Entropy 계산이 디폴트이기때문에 차원변환하면 안된다.
y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.int64).to(DEVICE)

print(y_train)

print("=================================================")
print(x_train.dtype)                # torch.float32
print(x_train.shape, y_train.shape) # torch.Size([105, 4]) torch.Size([105])
print(type(x_train))                # <class 'torch.Tensor'>
print("=================================================")

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
    # nn.Softmax(),
).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()   # Sparse Categorical Entropy + Softmax : y 원핫 인코딩X, 출력레이어 Softmax 제외
optimizer = optim.Adam(model.parameters(), lr=0.01)

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
    print('epochs: {}, loss: {}'.format(epoch, loss))   # verbose
print("==============================================")

# 4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        # loss2 = criterion(y, y_predict)
        # output이 앞이고 target이 뒤여야함.
        loss2 = criterion(y_predict, y)
        
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print('최종 loss :', last_loss)

y_predict = model(x_test)
print(type(y_predict))     # <class 'torch.Tensor'>
print(y_predict.size())    # torch.Size([45, 3])
print(y_test.size())       # torch.Size([45])

# 텐서를 넘파이배열로 변환
y_predict = y_predict.detach().cpu().numpy()
y_true = y_test.detach().cpu().numpy()
print(y_true.shape)        # (45,)

# y_predict argmax적용
y_predict = np.argmax(y_predict, axis=1)
print(y_predict.shape)     # (45,)

print('accuracy_score :', accuracy_score(y_true, y_predict))
# accuracy_score : 0.9555555555555556

"""
[TF, Sklearn, Pytorch 주요차이점 1 : (손실)함수 파라미터 순서]
 - TF, Sklearn : target, output 순
 - Pytorch : output, target 순

 [TF와 Pytorch 주요차이점 2 : 타겟라벨(y)전처리]
 - TF : y가 문자열이면 Label Encoding 등 정수 인코딩이 먼저 필요.
       - 손실 함수로 categorical_crossentropy 사용 시: y를 원-핫 인코딩 해야 함.
       - 손실 함수로 sparse_categorical_crossentropy 사용 시: y를 정수 인덱스 그대로 사용 (원-핫 인코딩 불필요).
 - PyTorch : 다중분류 시,
       - y는 반드시 0부터 시작하는 정수 인덱스(torch.long 타입)여야 함 (Label Encoding 필요)
       - 손실 함수로 nn.CrossEntropyLoss 사용 시: 
            -> 내부적으로 Softmax를 포함하므로, 출력층에는 활성화 함수를 사용X
            -> 내부적으로 sparse_categorical_crossentropy를 포함하므로, y 원-핫인코딩 사용X
 
 [타겟라벨의 종류에 따른 손실함수와 텐서타입]
 - 회귀 : MSELoss, torch.float32
 - 이진 : BCELoss, torch.float32
 - 다중 : CrossEntropyLoss, torch.long(=torch.int64)
"""
# =========================================================
# 파이토치 손실함수별 파라미터 순서 변경 시 영향 정리
#
# ┌────────────┬──────────────────────────────────────────────────────────────────────────────┬────────────┬────────────┬───────────────────────────────────────────────────────────────┐
# │ 손실함수   │ 공식(수식)                                                                   │ 논리오류   │ 수식오류   │ 설명                                                          │
# ├────────────┼──────────────────────────────────────────────────────────────────────────────┼────────────┼────────────┼───────────────────────────────────────────────────────────────┤
# │ MSELoss    │ MSE(output, target) = (1/N)∑(output_i - target_i)^2                          │   X        │    X       │ 제곱 연산은 대칭 ⇒ 순서 바뀌어도 결과 같음                             │
# ├────────────┼──────────────────────────────────────────────────────────────────────────────┼────────────┼────────────┼───────────────────────────────────────────────────────────────┤
# │ BCELoss    │ BCE(p, t) = -(1/N)∑[ t_i*log(p_i) + (1 - t_i)*log(1 - p_i) ]                  │   O        │    X       │ 입력/타겟 역할이 다름. 순서 바뀌면 손실 해석 오류                         │
# │            │                                                                              │            │            │ (계산은 됨, 값은 무의미)                                             │
# ├────────────┼──────────────────────────────────────────────────────────────────────────────┼────────────┼────────────┼───────────────────────────────────────────────────────────────┤
# │ CrossEntropy│ CE(x, c) = -log( softmax(x)_c ) = -log( exp(x_c) / ∑_j exp(x_j) )         │   O        │    O       │ x: logits(float, 2D), c: 인덱스(long, 1D).                        │
# │   Loss      │ (내부적으로 softmax 연산 포함)                                               │            │            │ 순서 바꾸면 내부 softmax에 정수연산 시도 → 즉시 에러 발생                    │
# └────────────┴──────────────────────────────────────────────────────────────────────────────┴────────────┴────────────┴───────────────────────────────────────────────────────────────┘
#
# ▷ 반드시 criterion(output, target) 순서로 사용할 것!
# ▷ CrossEntropyLoss는 수식에 softmax 연산이 포함되어 있어, 모델 출력에는 softmax를 쓰지 않음.
# =========================================================


