import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(5, 5)

    def forward(self, inputs):
        output = self.linear(inputs)
        return output

model = Model()

#
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

#
# 假设就一个数据
data = torch.rand([16, 10, 5])

# 前向计算要求数据都放进GPU0里面
# device = torch.device('cuda:0')
# data = data.to(device)
data = data.cuda()
inputs = data
target = 2 * data + 1

# 将网络同步到多个GPU中
model_p = torch.nn.DataParalle(model.cuda(), device_ids=[0, 1])
logits = model_p(inputs)


# # 接下来计算loss
# loss = crit(logits, target)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()