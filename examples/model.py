import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):

    def __init__(self, dimensions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dimensions[0], dimensions[1], bias=False)
        self.fc2 = nn.Linear(dimensions[1], dimensions[2])
        self.fc1.weight.data.normal_(mean=0.0, std=1.0)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten the input
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        # x = func.log_softmax(x, dim=1)
        return x
