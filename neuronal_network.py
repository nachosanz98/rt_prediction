import torch
from torch import nn
import torch.nn.functional as F

class NeuronalNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super(NeuronalNetwork, self).__init__()

        self.layer1 = nn.Linear(num_inputs, num_hiddens)
        self.layer2 = nn.Linear(num_hiddens, num_outputs)
    
    def forward(self, X):
        X = X.view(-1, self.layer1.in_features)
        H = F.relu(self.layer1(X))
        output = self.layer2(H)
        return output
