import torch
from .base_sampler import Sampler
from rectified_flow.rectified_flow import RectifiedFlow
from typing import Callable


class CurvedEulerSampler(Sampler):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
    ):
        super().__init__(
            rectified_flow, 
            num_steps, 
            time_grid, 
            record_traj_period, 
            callbacks, 
            num_samples,
        )

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
