import torch
import inspect

from torchvision import models
from gpu_men_track import MemTracker # 引用显存跟踪代码

device = torch.device('cuda:0')

frame = inspect.currentframe()     
gpu_tracker = MemTracker(frame)      # 创建显存检测对象

gpu_tracker.track()                  # 开始检测

cnn = models.vgg19(pretrained=True).to(device)
gpu_tracker.track()
# 上方为之前的代码

# 新增加的tensor
dummy_tensor_1 = torch.randn(30, 3, 512, 512).float().to(device)  # 30*3*512*512*4/1000/1000 = 94.37M
dummy_tensor_2 = torch.randn(40, 3, 512, 512).float().to(device)  # 40*3*512*512*4/1000/1000 = 125.82M
dummy_tensor_3 = torch.randn(60, 3, 512, 512).float().to(device)  # 60*3*512*512*4/1000/1000 = 188.74M

gpu_tracker.track()   # 再次打印
