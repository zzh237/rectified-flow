import torch
from .base_sampler import Sampler


class CurvedEulerSampler(Sampler):
    def step(self, **model_kwargs):
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.get_velocity(**model_kwargs)
        dtype = v_t.dtype
        x_t = x_t.to(torch.float32)
        v_t = v_t.to(torch.float32)

        self.rectified_flow.interp.solve(t, x_t=x_t, dot_x_t=v_t)
        x_1_pred = self.rectified_flow.interp.x_1
        x_0_pred = self.rectified_flow.interp.x_0

        # interplate to find x_{t_next}
        self.rectified_flow.interp.solve(t_next, x_0=x_0_pred, x_1=x_1_pred)
        self.x_t = self.rectified_flow.interp.x_t
        self.x_t = self.x_t.to(dtype)
