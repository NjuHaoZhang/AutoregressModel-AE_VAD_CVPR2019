import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# 参数和数据加载
input_size = 5
output_size = 2

batch_size = 30
data_size = 10000000

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("  In Model: input size", input.size(),
              "output size", output.size())

        return output

#
device_ids = [0,1]

model = Model(input_size, output_size)
model = nn.DataParallel(model,device_ids=device_ids) #
model = model.cuda(device_ids[0])

#
for data in rand_loader:
    if torch.cuda.is_available():
        input_var = Variable(data.cuda(device_ids[0]))
    else:
        input_var = Variable(data)
    output = model(input_var)
    print("Outside: input size", input_var.size(),
          "output_size", output.size())