import json 
from torch import nn, optim
from datetime import datetime

def load_model_architecture(json_file):
  """
  example of json_file: 
        {"criterion": "nn.MSELoss()",
         "optimizer":{"class": "optim.Adam", "learning_rate":0.0001, "weight_decay":0.8},
         "activation":{"h1": "nn.LeakyReLU()", "h2": "nn.LeakyReLU()", "h3": "nn.LeakyReLU()", "h4": "nn.LeakyReLU()" , "h5": "nn.LeakyReLU()", "h6": "nn.LeakyReLU()"},
         "node_size": {"h1": 512, "h2": 256, "h3": 128, "h4": 64, "h5": 32, "h6": 16}}
  """
    with open(json_file, 'r') as f:
      data = json.load(f)

    hidden_layers = []
    layers = zip(data['activation'].items(), data['node_size'].items())
    criterion = eval(data['criterion'])  # convert string to instance
    optimizer_config = data['optimizer']
    optimizer_config['class'] = eval(optimizer_config['class']) # convert string to instance

    for activation, size in layers:
        layer_name = activation[0]   # just for reference
        activation_func = eval(activation[1])
        size = size[1]
        hidden_layers.append((size, activation_func))

    return hidden_layers, criterion, optimizer_config

def write_model_architecture(file_name, criterion, model):
   with open(file_name, 'a') as f:
      time_stamp = datetime.now().strftime("%Y-%m-%d_%I%M%S_%p")
      optimizer_config = [(item, value) for item, value in model.optimizer.param_groups[0].items() if item != 'params']
      f.writelines(f'TimeStamp: {time_stamp}\n')
      f.writelines(f'criterion: {criterion.__class__.__name__}\n')
      f.writelines(f'optimizer_config: {optimizer_config}\n')
      f.writelines(f'model: {model}\n\n')

def write_train_history(file_name, line):
   with open(file_name, 'a') as f:
    f.writelines(f'{line}\n')
