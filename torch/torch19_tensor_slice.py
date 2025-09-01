import numpy as np
import pandas as pd
import torch
import random 
seed = 2497
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

x=torch.tensor([
    [[1,2],[3,4],[5,6]],
    [[7,8],[9,10],[11,12]]
])   

print(x[1,1,0])

print(x[:,-1,:])