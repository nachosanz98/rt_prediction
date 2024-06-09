import torch
from torch import nn
import torch.nn.functional as F

class NeuronalNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super(NeuronalNetwork, self).__init__()
        self.layer1 = nn.Linear(num_inputs, num_hiddens)
        self.batch_norm1 = nn.BatchNorm1d(num_hiddens)
        self.layer2 = nn.Linear(num_hiddens, num_hiddens)
        self.batch_norm2 = nn.BatchNorm1d(num_hiddens)
        self.layer3 = nn.Linear(num_hiddens, num_hiddens)
        self.batch_norm3 = nn.BatchNorm1d(num_hiddens)
        self.layer4 = nn.Linear(num_hiddens, num_outputs)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, X):
        X = X.view(-1, self.layer1.in_features)
        H1 = torch.relu(self.batch_norm1(self.layer1(X)))
        H1 = self.dropout(H1)
        H2 = torch.relu(self.batch_norm2(self.layer2(H1)))
        H2 = self.dropout(H2)
        H3 = torch.relu(self.batch_norm3(self.layer3(H2)))
        output = self.layer4(H3)
        return output
