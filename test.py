import torch
from main import Rockmate
from torchvision.models import resnet101, vgg11
from rkgb.main import *
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)
device = torch.device("cuda")
# model = resnet101().to(device)
model = vgg11().to(device)
optimizer = torch.optim.Adam(model.parameters())
x,y = torch.load('data/test_tensor_input_output.pt')
loss_func = torch.nn.CrossEntropyLoss()

# x = torch.randn([100, 3, 128, 128]).to(device)
m_budget = 2 * 1024**3 # 2GB
rk_resnet = Rockmate(model, x, m_budget)
y1 = rk_resnet(x)
loss = loss_func(y1, y1)
loss.backward()
rk_resnet.backward()
optimizer.step() # parameters in resnet are updated


# print_all_graphs(rkgb_results, name="vgg11", render_format="pdf", open=False)