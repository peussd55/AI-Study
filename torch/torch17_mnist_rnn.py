### <<48>>

import torch
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
from torch.utils.data import TensorDataset, DataLoader 
from torchvision.datasets import MNIST
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

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 device :', DEVICE)

data_list = ['regression', 'binary_classification', 'multiclass_classification']
data_type = 'multiclass_classification'
########################## 랜덤고정 & 쿠다확인 & 데이터종류명시 ##########################

# 1. 데이터
# 전처리파이프라인 구성
import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5), (0.5))])     # 표준화 : (x-0.5) / 0.5
# Resize(0) : 0x0로 리사이즈
# ToTensor() : 토치텐서타입으로 바꾸기 + MinMaxScaler
# MinMaxScaler를 할지 StandardScaler를 할지 정하는 방법 : 통상적으로 활성화함수를 ReLU(0이상)를 쓸때는 MinMaxScaler를 적용하고 tanh(-1~1)을 쓸때는 StandardScaler적용
#################### tr.Normailze((0.5), (0.5)) ####################
# z_score Normalizeation (정규화의 표준화)
# (x-평균) / 표준편차
# (x - 0.5) / 0.5   위 식처럼해야하는데 통상 평균 0.5, 표편 0.5로 계산하면 
# -1 ~ 1 사이의 범위가 나오니 이미지 전처리에서는 통상 0.5 0.5 한다.
####################################################################

# 데이터로드
path = './_data/torch/'
train_dataset = MNIST(path, train=True, download=True, transform=transf)  # train=True 학습데이터 로드
test_dataset = MNIST(path, train=False, download=True, transform=transf)
print(type(train_dataset))              # <class 'torchvision.datasets.mnist.MNIST'>
print(train_dataset[0])
# 전처리를 거쳐서 (PIL객체, 레이블) -> (텐서객체, 레이블)로 반환
# (tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          ...,
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.]]]), 5)
print(type(train_dataset[0]))           # <class 'tuple'>
print(type(train_dataset[0][0]))        # <class 'torch.Tensor'> (transform적용) 또는 <class 'PIL.Image.Image'>
print(type(train_dataset[0][1]))        # <class 'int'>

# 전처리확인
img_tensor, label = train_dataset[0]
print(img_tensor.shape)                     # x : torch.Size([1, 56, 56]). torch데이터는 채널이 앞에 와야함.
print(label)                                # y : 5
print(len(train_dataset.classes))           # 라벨갯수 : 10
print(img_tensor.min(), img_tensor.max())   # sacler : tensor(-1.) tensor(0.9843)
# train_dataset.data, train_dataset.target : transform 되지않은 원래의 데이터셋을 불러온다.

# # 스케일링 (transform 적용시 불필요)
# x_train, y_train = train_dataset.data/255, train_dataset.targets
# x_test, y_test = test_dataset.data/255, test_dataset.targets
# print(np.min(x_train.numpy()), np.max(x_train.numpy()))     # 0.0 1.
# print(x_train.shape, y_train.shape)                         # torch.Size([60000, 28, 28]) torch.Size([60000])
# print(x_train.shape, y_train.shape)                         # torch.Size([60000, 28, 28]) torch.Size([60000])

# # reshape (DNN모델 구성시 필요)
# x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 784)

# # train val 분리 (transform 적용시 사용불가 : train_dataset을 .data(x), .target(y)로 직접분리했을시 사용)
# val_size = 0.1
# x_train, x_val, y_train, y_val = train_test_split(  
#     x_train, y_train, test_size=val_size, shuffle=True, random_state=SEED,
#     stratify=y_train if data_type != 'regression' else None,
# )

# train, val 분리 (transform 적용시 사용)
from torch.utils.data import random_split
total_size = len(train_dataset)
val_size = int(total_size * 0.1)
train_size = total_size - val_size
train_set, val_set = random_split(train_dataset, [train_size, val_size])
print(f"분리된 훈련 데이터셋: {len(train_set)}개")
print(f"분리된 검증 데이터셋: {len(val_set)}개")
# 분리된 훈련 데이터셋: 54000개
# 분리된 검증 데이터셋: 5000개

# # shape 확인 (transform 적용시 사용불가 : train_dataset을 .data(x), .target(y)로 직접분리했을시 사용)
# # .to(DEVICE)는 모델구현부분에서 직접명시 / .to(DEVICE) 의미 : 데이터를 gpu메모리로 전송
# print("==================== 최종 shape =====================")
# print(x_train.shape, y_train.size())
# print(x_val.shape, y_val.size())
# print(x_test.shape, y_test.size())
# print(type(x_train), type(y_train))
# # torch.Size([54000, 784]) torch.Size([54000])
# # torch.Size([6000, 784]) torch.Size([6000])
# # torch.Size([10000, 784]) torch.Size([10000])
# # <class 'torch.Tensor'> <class 'torch.Tensor'>

# # 토치데이터셋 생성 (transform 적용시 사용불가 : train_dataset을 .data(x), .target(y)로 직접분리했을시 사용)
# # : x,y 합치기(tuple 형태로), intput으로 텐서받아야함
# train_set = TensorDataset(x_train, y_train)
# val_set = TensorDataset(x_val, y_val)
# test_set = TensorDataset(x_test, y_test)
# print(type(train_set))      
# # <class 'torch.utils.data.dataset.TensorDataset'>

# 하이퍼파라미터 설정
# input_dim = x_train.shape[1]
# output_dim = len(np.unique(y_train.numpy()))
input_channel = train_dataset[0][0].shape[0]
output_dim = len(train_dataset.classes)
lr=1e-4
epochs = 10
criterion_map = {
    'regression': nn.MSELoss,
    'binary_classification': nn.BCELoss,
    'multiclass_classification': nn.CrossEntropyLoss
}
criterion = criterion_map[data_type]()
batch_size = 32
timesteps=3

# 토치데이터로더 생성
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)       # shuffle=True : 매 epoch마다 배치순서 섞음. 과적합방지 필수
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)          # shuffle=False : 재현성(추적용이)보장
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # shuffle=False : 재현성(추적용이)보장

# (하이퍼)파라미터 & 데이터 정보 출력
print("==================== (하이퍼)파라미터 & 데이터 정보 출력 =====================")
# print("[파라미터] input_dim :", input_dim)
# print("[파라미터] output_dim :", output_dim)
print("[파라미터] input_channel :", input_channel)
print("[파라미터] output_dim :", output_dim)
print("----------------")
print("[하이퍼 파라미터] epochs :", epochs)
print("[하이퍼 파라미터] criterion :", 'nn.' + str(criterion))
print("[하이퍼 파라미터] 배치사이즈 :", batch_size)
print("[하이퍼 파라미터] learning_rate :", lr)
print("[하이퍼 파라미터] timesteps :", timesteps)
print("----------------")
print("원시데이터 갯수 :", len(train_dataset)+len(test_dataset))
print("train 갯수 :", len(train_set))
print("validation 갯수 :", len(val_set))
print("test 갯수 :", len(test_dataset))
print("배치 갯수 :", len(train_loader))
# [파라미터] input_channel : 1
# [파라미터] output_dim : 10
# ----------------
# [하이퍼 파라미터] epochs : 10
# [하이퍼 파라미터] criterion : nn.CrossEntropyLoss()
# [하이퍼 파라미터] 배치사이즈 : 32
# [하이퍼 파라미터] learning_rate : 0.0001
# ----------------
# 원시데이터 갯수 : 70000
# train 갯수 : 54000
# validation 갯수 : 6000
# test 갯수 : 10000
# 배치 갯수 : 1688

# exit()
# 2. 모델
class RNN(nn.Module):
    def __init__(self, input_channel, timesteps):
        super().__init__()
        # super(RNN, self).__init__()
        self.rnn_layer1 = nn.RNN(
            input_size=input_channel,
            hidden_size=32,
            # num_layers=1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(timesteps*32, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):   # 오버라이딩
        
        # cnn데이터 4차원 - > 3차원으로 변경
        # 입력 x shape: (batch, channel, height, width) -> (32, 1, 56, 56)
        # 1. RNN에 맞는 3D 형태로 변경: (batch, seq_len, input_size)
        #    채널 차원을 제거(squeeze)하여 (batch, height, width)로 만듦
        x = x.squeeze(1)  # [1]인덱스제거. 결과 x shape: (32, 56, 56)
        
        x, _ = self.rnn_layer1(x)
        x = self.relu(x)
        
        x = x.reshape(-1, timesteps*32)     # timesteps * feature
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x
        
model = RNN(input_channel, timesteps).to(DEVICE)

# summary 출력
from torchsummary import summary
summary(model, (batch_size, timesteps, input_channel))    # 입력크기를 직접 전부 다 명시

exit()
# 3. 컴파일, 훈련
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
all_true_labels = []    # 실제 정답 라벨을 담을 리스트 생성
with torch.no_grad():   # 배치단위로 예측
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)
        # GPU상에서 loss계산없으므로 y_batch는 .to(DEVICE) 필요X
        
        # 모델 예측
        y_predict_batch = model(x_batch)
        
        # 예측값과 실제 정답값을 각각 cpu로 보내서 리스트에 추가
        all_predictions.append(y_predict_batch.detach().cpu())   # [텐서1, 텐서2, 텐서3, ...] / y_predict_batch는 gpu메모리에서 생성되었기때문에 .cpu()로 옮김
        all_true_labels.append(y_batch.detach().cpu())           # y_batch는 gpu메모리에 있지않기때문에 .cpu()를 하지 않아도되나, 명확성을 위해 통일함

# 텐서를 넘파이배열로 변환(output, target)
y_predict = torch.cat(all_predictions, dim=0).numpy()   # torch.cat : 행단위로 텐서1 + 텐서2 + 텐서3 + ... 합한 후 numpy로 변환
y_true = torch.cat(all_true_labels, dim=0).numpy()

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

"""