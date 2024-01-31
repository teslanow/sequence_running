import torch
import numpy as np
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)
x = torch.randn([100, 3, 128, 128])
torch.save(x, 'data/test_tensor_input.pt')