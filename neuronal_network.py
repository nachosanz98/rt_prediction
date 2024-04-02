import torch
from torch import nn
import torch.nn.functional as F

class NeuronalNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super(NeuronalNetwork, self).__init__()

        self.hparams = {
            "num_inputs": num_inputs,
            "num_outputs": num_outputs,
            "num_hiddens": num_hiddens,
            "lr": lr,
            "sigma": sigma
        }

        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma).float()
        self.b1 = nn.Parameter(torch.zeros(num_hiddens)).float()
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma).float()
        self.b2 = nn.Parameter(torch.zeros(num_outputs)).float()
    
    def forward(self, X):
        X = X.reshape((-1, self.hparams["num_inputs"])).float()
        H = F.relu(torch.matmul(X, self.W1) + self.b1)
        return torch.matmul(H, self.W2) + self.b2
    
