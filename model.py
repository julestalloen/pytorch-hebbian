import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(params['input_size'], params['hidden_units']),
        torch.nn.ReLU(),
        torch.nn.Linear(params['hidden_units'], params['output_size']),
    )
    """

    def __init__(self, dimensions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dimensions[0], dimensions[1], bias=False)
        self.fc2 = nn.Linear(dimensions[1], dimensions[2])

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten the input
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x
