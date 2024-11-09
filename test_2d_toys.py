#%% 
import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import sympy
import matplotlib.pyplot as plt
from collections import namedtuple

#from rectified_flow.rectified_flow import AffineInterp,  RectifiedFlow
#from rectified_flow.samplers import EulerSampler, OverShootingSampler, SDESampler
from rectified_flow import * 
import rectified_flow.toys  as toys 


# generate toy data
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)

n_samples = 1000
pi0 = dist.MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
#pi1 = toys.create_circular_gmm(6, radius=10, dim=2, device=device)
pi1 = toys.create_two_point_gmm(x=10.0, y=5, std=0.0001, device=device)
D0 = pi0.sample([n_samples]).to(device)
D1 = pi1.sample([n_samples]).to(device)

model = toys.MLPWithTimeInput(2, hidden_sizes = [128,128]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = lambda vt, dot_xt, xt, t, wts: torch.mean(wts * (vt - dot_xt)**2)
rf = RectifiedFlow(
    D0 = D0,
    D1 = D1,
    model = model,
    optimizer = optimizer,
    interp = AffineInterp('straight'),
    device = device,
    pi0 = pi0,
    seed = 1,
    criterion = criterion,
    )

rf.train(num_iterations=1000, batch_size=1000)

sampler_seed = 43

euler_sampler = EulerSampler(rf, num_steps = 100, num_points=1000, seed=sampler_seed)
euler_sampler.sample()

o_sampler = OverShootingSampler(rf, num_steps=100, c=10.0, overshooting_method='t+dt', num_points = 1000, seed=sampler_seed)
o_sampler.sample()

sde_sampler = SDESampler(rf, num_steps=100,  e=lambda t: 10*t, num_points=1000, seed=sampler_seed)
sde_sampler.sample()

plt.figure(figsize=(10*3,3)); kk = 1; KK= 10
plt.subplot(1,KK,kk); kk+=1
rf.plot_loss_curve()

plt.subplot(1,KK,kk); kk+=1
euler_sampler.plot_2d_results()

plt.subplot(1,KK,kk); kk+=1
o_sampler.plot_2d_results()

plt.subplot(1,KK,kk); kk+=1
sde_sampler.plot_2d_results()

# %%
