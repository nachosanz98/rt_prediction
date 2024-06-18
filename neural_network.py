import torch
from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens_list):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        input_size = num_inputs
        for hidden in num_hiddens_list:
            self.layers.append(nn.Linear(input_size, hidden))
            self.batch_norms.append(nn.BatchNorm1d(hidden))
            input_size = hidden

        self.output_layer = nn.Linear(input_size, num_outputs)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, X):
        X = X.view(-1, self.layers[0].in_features)
        
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            X = torch.relu(batch_norm(layer(X)))
            X = self.dropout(X)
        
        output = self.output_layer(X)
        return output
