import torch.nn as nn
import torch.nn.functional as F

# Define fully-connected  neural network
class QNetwork(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, hidden_dims=[64, 64], act=F.relu):
        super(QNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        self.act = act
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
          
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x) # Do not apply activation to the last layer
        return x