import torch
import torch.nn as nn

"""

  Initialization functions for the DhoPINN model. 
  These can be passed to the model via the init_factory argument to customize the weight initialization of the neural network layers. 
  Proper initialization can help with training convergence and performance.

"""

def xavier_init(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    nn.init.zeros_(m.bias)

def orthogonal_init(m):
  if isinstance(m, nn.Linear):
    nn.init.orthogonal_(m.weight)
    nn.init.zeros_(m.bias)
