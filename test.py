import torch
from main import Rockmate
from torchvision.models import resnet101, vgg11
from rkgb.main import *

device = torch.device("cuda")
# model = resnet101().to(device)
model = vgg11().to(device)
x = torch.randn([100, 3, 128, 128]).to(device)
m_budget = 2 * 1024**3 # 2GB
rk_resnet = Rockmate(model, x, m_budget, solve=False)
rk_resnet(x)


# print_all_graphs(rkgb_results, name="vgg11", render_format="pdf", open=False)