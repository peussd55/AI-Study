from torchvision.datasets import MNIST
import torch
from torch.utils.data import Dataset, DataLoader

import random

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print("torch:", torch.__version__,'사용:', DEVICE)

#################랜덤고정###################
seed=222
random.seed(seed) # 넘파이 랜검 고정 
torch.manual_seed(seed) # 토치 랜덤고정
torch.cuda.manual_seed(seed)  #토치 쿠다 시드 고정
##########################################

path ='./_data/torch/'

train_dataset = MNIST(path, train=True, download=True)
test_dataset = MNIST(path, train=False, download=True)

# x_train, y_train= train_dataset.data/255., train_dataset.targets
# x_test, y_test= train_dataset.data/255., train_dataset.targets

print(train_dataset)
# exit()
class Mydataset(Dataset):
    def __init__(self,dataset):
        self.x=dataset.data/255.
        self.y=dataset.targets
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return torch.tensor(self.x[index]), torch.tensor(self.y[index])
    
train_dataset=Mydataset(train_dataset)
test_dataset=Mydataset(test_dataset)

train_loader=DataLoader(train_dataset, batch_size=2,shuffle=2)
test_loader=DataLoader(train_dataset, batch_size=2,shuffle=2)
#4.출력

for batch_idx,(xb,yb) in enumerate(train_loader):
    print("배치:", batch_idx)
    print("x:배치",xb )
    print("Y:배치",yb )
    
    if batch_idx == 2: 
        break