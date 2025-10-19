### <<47>>

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
random.seed(SEED)               # 파이썬 랜덤 고정
np.random.seed(SEED)            # 넘파이 랜덤 고정
torch.manual_seed(SEED)         # 토치 랜덤 고정
torch.cuda.manual_seed(SEED)    # 토치 쿠다 랜덤 고정 

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)

data_list = ['regression', 'binary_classification', 'multiclass_classification']
data_type = 'multiclass_classification'
########################## 랜덤고정 & 쿠다확인 & 데이터종류명시 ##########################

# 1. 데이터
# (최초실행시 로컬다운)
path = './_data/torch/'
train_dataset = CIFAR100(path, train=True, download=True)
test_dataset = CIFAR100(path, train=False, download=True)
print(type(train_dataset))          # <class 'torchvision.datasets.cifar.CIFAR100'>
print(train_dataset[0])             # (<PIL.Image.Image image mode=RGB size=32x32 at 0x20BB3D1B5E0>, 6)
# tuple : ()로 감싸고 , 로 요소를 구분

# 스케일링
x_train, y_train = train_dataset.data/255, train_dataset.targets
x_test, y_test = test_dataset.data/255, test_dataset.targets
print(np.min(x_train), np.max(x_train))     # 0.0 1.0
print(type(x_train), type(y_train))         # <class 'numpy.ndarray'> <class 'list'> -> 텐서변환해줘야함
print(x_train.shape, len(y_train))          # (50000, 32, 32, 3) 50000

# reshape
x_train, x_test = x_train.reshape(-1, 32*32*3), x_test.reshape(-1, 32*32*3)     # nparray라 reshape써야함.

# train val 분리
# train_test_split : torch.Tensor 타입도 분리해줌
val_size = 0.1
x_train, x_val, y_train,y_val = train_test_split(
    x_train, y_train, test_size=val_size, shuffle=True, random_state=SEED,
    stratify=y_train if data_type != 'regression' else None,
)

# 텐서변환
# .to(DEVICE)는 모델구현부분에서 직접명시 / .to(DEVICE) 의미 : 데이터로 gpu메모리로 전송
x_train = torch.tensor(x_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
if data_type == 'multiclass_classification':
    # 다중 분류: 1D, long 타입으로 변환 (unsqueeze 없음)
    y_train, y_val, y_test = [
        torch.tensor(v, dtype=torch.long) for v in [y_train, y_val, y_test]
    ]
else:
    # 회귀 또는 이진 분류: 2D (unsqueeze 적용), float 타입으로 변환
    y_train, y_val, y_test = [
        torch.tensor(v, dtype=torch.float32).unsqueeze(1) for v in [y_train, y_val, y_test]
    ]
    
# shape 확인
print("==================== 최종 shape ====================")
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)
print(type(x_train), type(y_train))
# torch.Size([45000, 3072]) torch.Size([45000])
# torch.Size([5000, 3072]) torch.Size([5000])
# torch.Size([10000, 3072]) torch.Size([10000])
# <class 'torch.Tensor'> <class 'torch.Tensor'>
# <class 'torch.utils.data.dataset.TensorDataset'>

# 토치데이터셋 생성 : x, y 합치기(tuple 형태로), intput으로 텐서받아야함
train_set = TensorDataset(x_train, y_train)
val_set = TensorDataset(x_val, y_val)
test_set = TensorDataset(x_test, y_test)
print(type(train_set))      # <class 'torch.utils.data.dataset.TensorDataset'>

# 하이퍼파라미터 설정
input_dim = x_train.shape[1]
output_dim = len(np.unique(y_train.numpy()))
lr = 1e-4
epochs = 100
criterion_map = {
    'regression' : nn.MSELoss,
    'binary_classification' : nn.BCELoss,
    'multiclass_classification' : nn.CrossEntropyLoss
}
criterion = criterion_map[data_type]()
batch_size = 32

# 토치데이터로더 생성 : 토치데이터셋을 섞고 배치사이즈만큼 분리해
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)   # shuffle=True : 매 epoch마다 배치순서 섞음. 과적합방지 필수
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)      # shuffle=False : 재현성(추적용이)보장
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)    # shuffle-False : 재현성(추적용이)보장
print(type(train_loader))   # <class 'torch.utils.data.dataloader.DataLoader'>

# (하이퍼)파라라미터 & 데이터 정보 출력
print("==================== (하이퍼)파라미터 & 데이터 정보 출력 ====================")
print("[파라미터] input_dim :", input_dim)
print("[파라미터] output_dim :", output_dim)
print("----------------")
print("[하이퍼 파라미터]  :", lr)
print("[하이퍼 파라미터] epochs :", epochs)
print("[하이퍼 파라미터] criterion :", 'nn.' + str(criterion))
print("[하이퍼 파라미터] 배치사이즈 :", batch_size)
print("[하이퍼 파라미터] learning_rate :", lr)
print("----------------")
print("원시데이터 갯수 :", len(train_dataset)+len(test_dataset))
print("train 갯수 :", len(x_train))
print("validation 갯수 :", len(x_val))
print("test 갯수 :", len(x_test))
print("배치 갯수 :", len(train_loader))
# [파라미터] input_dim : 3072
# [파라미터] output_dim : 10
# ----------------
# [하이퍼 파라미터] epochs : 100
# [하이퍼 파라미터] criterion : nn.CrossEntropyLoss()
# [하이퍼 파라미터] 배치사이즈 : 32
# [하이퍼 파라미터] learning_rate : 0.0001
# ----------------
# 원시데이터 갯수 : 60000
# train 갯수 : 45000
# validation 갯수 : 5000
# test 갯수 : 10000
# 배치 갯수 : 1407

# exit()
# 2. 모델
class DNN(nn.Module):
    def __init__(self, num_features, output_dim):
        super().__init__()
        # super(DNN, self).__init__()
        
        # 히든레이어 Sequential() 구현
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.output_layer = nn.Linear(32, output_dim)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.output_layer(x)
        
        return x

model = DNN(input_dim, output_dim).to(DEVICE)

# 3.컴파일, 훈련
criterion = criterion
optimizer = optim.Adam(model.parameters(), lr=lr)

# 훈련함수 정의
def train(model, criterion, optimizer, loader):
    model.train()   # Dropout, BatchNorm 등의 레이어를 활성화. 디폴트이지만 2epoch부터 eval()로 남아있는 것 방지 위해 명시해줘야함.
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()   # 기울기 0으로 초기화
        
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()         # 기울기(gradient)값까지만 계산 : ∂L/∂w
        optimizer.step()        # 가중치갱신 : w = w - lr * ∂L/∂w
        
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss/len(loader), epoch_acc/len(loader)

# 검증(평가)함수 정의
def evaluate(model, criterion, loader):
    model.eval()    # Dropout, BatchNorm 등의 레이어를 비활성화
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():   # 기울기계산기록X (메모리폭증, 불필요한 연산제거)
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            
            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss/len(loader), epoch_acc/len(loader)

# 훈련시작
EPOCH = epochs
start_time = time.time()
for epoch in range(1, EPOCH+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, val_loader)
    print(f'epoch : {epoch} | loss : {loss:.4f}, acc : {acc:.4f},\
        val_loss : {val_loss:.4f}, val_acc : {val_acc:.4f}'
          )
print("===============================================")
end_time = time.time()
print("걸린시간 :", end_time - start_time, "초") 

# 4.1 평가
loss, acc = evaluate(model, criterion, test_loader)
print("최종 loss :", loss)
print("최종 acc :", acc)

# 4.2 예측 (예측데이터가 없어서 테스트데이터로 대신사용)
# target 생성
model.eval()            # eval모드로 변환
all_predictions = []    # 예측결과를 담을 리스트생성
with torch.no_grad():   # 배치단위로 예측
    for x_batch, y_batch_dummy in test_loader:
        x_batch = x_batch.to(DEVICE)
        y_predict_batch = model(x_batch)
        all_predictions.append(y_predict_batch.cpu())   # [텐서1, 텐서2, 텐서3, ...]

# 텐서를 넘파이배열로 변환(output, target)
y_predict = torch.cat(all_predictions, dim=0).numpy()   # torch.cat : 행단위로 텐서1 + 텐서2 + 텐서3 + ... 합한 후 numpy로 변환
y_true = y_test.detach().cpu().numpy()

# target 변환
if data_type == 'binary_classification':
    # 1 또는 0 으로 변환 : astype은 True를 1, False를 0으로 반환
    y_predict = (y_predict > 0.5).astype(np.int32)  # 또는 np.round(y_predict)
elif data_type == 'multiclass_classification':
    y_predict = np.argmax(y_predict, axis=1)
else:
    y_predict = y_predict
    
# 예측스코어 확인
if data_type == 'regression':
    print('r2_score :', r2_score(y_true, y_predict))
else:
    print('acc :', accuracy_score(y_true, y_predict))

"""
걸린시간 : 399.58688521385193 초
최종 loss : 3.3107643462598513
최종 acc : 0.21904952076677317
acc : 0.2189
"""