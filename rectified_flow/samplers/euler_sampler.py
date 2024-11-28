import torch
from .base_sampler import Sampler


class EulerSampler(Sampler):
    def step(self, **model_kwargs):
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.get_velocity(**model_kwargs)
        dtype = x_t.dtype
        x_t = x_t.to(torch.float32)
        v_t = v_t.to(torch.float32)
        self.x_t = x_t + (t_next - t) * v_t
        self.x_t = self.x_t.to(dtype)