#@title toy data  & model
import torch
import torch.nn as nn
import torch.distributions as dist
from collections import namedtuple

# create toy gaussian mixture models
def create_circular_gmm(n_components, radius, dim=2, std = 1.0, device=None):
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    angles = torch.linspace(0, 2 * torch.pi, n_components+1)[:-1].to(device)
    means = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1).to(device)
    stds = std * torch.ones(n_components, dim).to(device)
    weights = torch.ones(n_components).to(device)
    weights /= weights.sum()
    gmm = dist.MixtureSameFamily(dist.Categorical(weights), dist.Independent(dist.Normal(means, stds), 1))
    return gmm

def create_two_point_gmm(x=10.0, y=10.0, std=1.0, device=None):
  if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
  means = torch.tensor([[x, y], [x, -y]]).to(device)
  n_components = 2
  n_features = 2
  stds = torch.ones(n_components, n_features).to(device) * std
  weights = torch.ones(n_components).to(device); weights /= weights.sum()
  gmm = dist.MixtureSameFamily(dist.Categorical(weights), dist.Independent(dist.Normal(means, stds), 1))
  return gmm

# generic MLP model
class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU, linear_layer=nn.Linear):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.linear_layer = linear_layer
        for i in range(len(layer_sizes) - 1):
            self.layers.append(self.linear_layer(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation after the last layer
                self.layers.append(self.activation())

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x

# a model for velocity field v(x,t)
class MLPWithTimeInput(nn.Module):
  def __init__(self, dim, hidden_sizes=[128, 128, 128], output_dim = None):
    super().__init__()
    if output_dim is None:
      output_dim = dim
    self.mlp = MLP([dim+1, *hidden_sizes, output_dim])
  def forward(self, x, t):
    t = t.squeeze()
    t = t.view(t.shape[0], -1)
    x = x.view(x.shape[0], -1)
    xt = torch.cat((x, t), dim=1)
    return self.mlp(xt)

