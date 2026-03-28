import torch
import torch.nn as nn
from typing import Callable, Optional

class DhoPINN(nn.Module):
	def __init__(self, hidden_layers: int = 3, hidden_dim: int = 64, activation: nn.Module = nn.Tanh(), init_factory: Optional[Callable[[nn.Module], None]] = None, x0: float = 0.7, v0: float = 1.2):
		super(DhoPINN, self).__init__()

		self.input_dim = 2
		self.x0 = x0
		self.v0 = v0

		layers = []
		layers.append(nn.Linear(self.input_dim, hidden_dim))
		layers.append(activation)

		for _ in range(hidden_layers-1):
			layers.append(nn.Linear(hidden_dim, hidden_dim))
			layers.append(activation)
        
		layers.append(nn.Linear(hidden_dim, 1))
		self.net = nn.Sequential(*layers)
	  
		if init_factory:
			self.apply(init_factory)
		else:
			self.default_init()

	def default_init(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				nn.init.constant_(m.bias, 0)
            
                    
	def forward(self, z, Xi):
		input_data = torch.cat((z, Xi), dim=1)
		output = self.net(input_data)

		hc_output = self.x0 + self.v0*z + 0.5*(z**2)*output

		return hc_output
  
