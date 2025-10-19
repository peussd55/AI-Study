### <<49>>

import torch
from torch.utils.data import Dataset, DataLoader

# 1. 커스텀 데이터셋 만들기
class MyDataset(Dataset):
    def __init__(self):
        self.x = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        self.y = [0, 1, 0, 1, 0]
        
    def __len__(self):
        return len(self.x)      # 어차피 x길이와 y길이는 같은 데이터로 훈련시키기때문에 self.x만 return한다.
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])
    
# 2. 인스턴스 생성
dataset = MyDataset()

# 3. DataLoader에 쏙
loader = DataLoader(dataset, batch_size=3, shuffle=True)

# 4. 출력
for batch_idx, (xb, yb) in enumerate(loader):
    print("========배치 :", batch_idx, "=========")
    print("x :", xb)
    print("y :", yb)
    
# ========배치 : 0 ==========
# x : tensor([[3.],
#         [2.],
#         [4.]])
# y : tensor([0, 1, 1])
# ========배치 : 1 ==========
# x : tensor([[5.],
#         [1.]])
# y : tensor([0, 0])