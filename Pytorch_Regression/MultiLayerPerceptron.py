import torch, tqdm, math
import torch.nn as nn, torch.optim as optim, torch.cuda as cuda
from torch.nn import Module

class MLP(nn.Module):
  def __init__(self, input_size, layers: list, optimizer_config, dropout_p):
    super(MLP, self).__init__()
    self.device = torch.device('cuda' if cuda.is_available() else 'cpu')
    self.layers = nn.ModuleList()
    self.input_size = input_size 
    self.dropout_p = dropout_p 

    for size, activation in layers:
      self.layers.append(nn.Linear(input_size, size))
      self.layers.append(nn.BatchNorm1d(size))

      input_size = size # for next layer

      if activation is not None:
        assert isinstance(activation, Module), "Each tuples should contain a size (int) and a torch.nn.modules.Module."
        self.layers.append(activation)
    
      self.layers.append(nn.Dropout(p=self.dropout_p))

    for layer in self.layers:
      if isinstance(layer, nn.Linear):
        #nn.init.xavier_uniform_(layer.weight)
        nn.init.kaiming_uniform_(layer.weight) # 'He' initialization for LeakyReLU() activation
        nn.init.zeros_(layer.bias)

    self.to(self.device)
    self.learning_rate = optimizer_config['learning_rate']
    self.weight_decay = optimizer_config['weight_decay']
    self.optimizer = optimizer_config['class'](params=self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

  def forward(self, input_data):
    for layer in self.layers:
      input_data = layer(input_data)
    return input_data
