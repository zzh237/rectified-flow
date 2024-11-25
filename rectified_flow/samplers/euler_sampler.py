import torch
from .base_sampler import Sampler


class EulerSampler(Sampler):
    def step(self, **model_kwargs):
        t, t_next, X_t = self.t, self.t_next, self.X_t
        v_t = self.get_velocity(**model_kwargs)
        dtype = X_t.dtype
        X_t = X_t.to(torch.float32)
        v_t = v_t.to(torch.float32)
        self.X_t = X_t + (t_next - t) * v_t
        self.X_t = self.X_t.to(dtype)