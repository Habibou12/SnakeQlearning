import torch
import numpy as np

class Qnetwork(torch.nn.Module):

    def __init__(self):
        super(Qnetwork, self).__init__()
        self.linear1 = torch.nn.Linear(11, 200)
        self.linear2 = torch.nn.Linear(200, 200)
        self.linear3 = torch.nn.Linear(200, 3)
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x=self.linear3(x)
        return x


