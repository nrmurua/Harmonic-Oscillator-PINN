import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import unittest
import torch
import torch.nn as nn
from src.dho_pinn import DhoPINN

class TestDhoPINN(unittest.TestCase):
  def setUp(self):
    self.x0, self.v0 = 0.7, 1.2
    self.model = DhoPINN(x0=self.x0, v0=self.v0)
    self.z_sample = torch.tensor([[0.0], [1.0], [2.0]], requires_grad=True)
    self.Xi_sample = torch.tensor([[0.1], [0.2], [0.3]], requires_grad=True)

  def test_hard_constraints(self):
      z_zero = torch.tensor([[0.0]], requires_grad=True)
      Xi = torch.tensor([[0.2]])

      pred = self.model(z_zero, Xi)
      self.assertAlmostEqual(pred.item(), self.x0, places=6, msg="Model does not satisfy initial position constraint at z=0")

      grad = torch.autograd.grad(pred, z_zero, create_graph=True)[0]
      self.assertAlmostEqual(grad.item(), self.v0, places=6, msg="Model does not satisfy initial velocity constraint at z=0")

  def test_factory_injection(self):
      def custom_init(m):
          if isinstance(m, nn.Linear):
              nn.init.constant_(m.weight, 0.5)
              nn.init.constant_(m.bias, 0.1)

      model_custom = DhoPINN(activation=nn.ReLU(),init_factory=custom_init, x0=self.x0, v0=self.v0)
      
      self.assertIsInstance(model_custom.net[1], nn.ReLU, "Custom activation function not set correctly")
      val = model_custom.net[0].weight.data[0,0].item()
      self.assertAlmostEqual(val, 0.5, places=6, msg="Custom weight initialization not applied correctly")
      bias_val = model_custom.net[0].bias.data[0].item()
      self.assertAlmostEqual(bias_val, 0.1, places=6, msg="Custom bias initialization not applied correctly")

  def test_gradient_flow(self):
    pred = self.model(self.z_sample, self.Xi_sample)

    dz = torch.autograd.grad(pred, self.z_sample, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
    self.assertIsNotNone(dz, "Gradient with respect to z is None")

    dzz = torch.autograd.grad(dz, self.z_sample, grad_outputs=torch.ones_like(dz), create_graph=True)[0]
    self.assertIsNotNone(dzz, "Second gradient with respect to z is None")

  def test_output_shape(self):
    batch_size = 32
    z = torch.randn(batch_size, 1)
    Xi = torch.randn(batch_size, 1)
    output = self.model(z, Xi)
    self.assertEqual(output.shape, (batch_size, 1), "Output shape is incorrect")

if __name__ == '__main__':
    unittest.main(verbosity=2)