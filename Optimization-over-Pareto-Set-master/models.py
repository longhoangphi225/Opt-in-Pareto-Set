import torch
from torch import nn

device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
class Hypernetwork(nn.Module):
  def __init__(self, ray_hidden_dim=100, out_dim=10, target_hidden_dim=50, n_hidden=1, n_tasks=2):
      super().__init__()
      self.n_hidden = n_hidden
      self.n_tasks = n_tasks
 
      self.ray_mlp = nn.Sequential(
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),

            # nn.Linear(ray_hidden_dim, ray_hidden_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(ray_hidden_dim, ray_hidden_dim),
            # nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, out_dim),
        )
      
      self.hidden_0_weights = nn.Linear(ray_hidden_dim, target_hidden_dim)
      self.hidden_0_bias = nn.Linear(ray_hidden_dim, target_hidden_dim)
      for j in range(n_tasks):
            setattr(self, f"task_{j}_weights", nn.Linear(ray_hidden_dim, target_hidden_dim * out_dim))
            setattr(self, f"task_{j}_bias", nn.Linear(ray_hidden_dim, out_dim))

  def forward(self, ray):
      features = self.ray_mlp(ray)
      return features.unsqueeze(0)