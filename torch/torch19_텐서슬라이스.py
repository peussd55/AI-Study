### <<49>>

import torch

x = torch.tensor([
    [[1,2], [3,4], [5,6]],
    [[7,8], [9,10], [11,12]]
])

print(x.shape)  # torch.Size([2, 3, 2]) : [N, H, W] : [배치, 행, 열] : [배치, 높이, 너비]

x = x[:, -1, :] # 배치, 열은 그대로. 맨마지막 행(=높이) 만 쓰겠다.
print(x.shape)  # torch.Size([2, 2])
print(x)
# tensor([[ 5,  6],
#         [11, 12]])

