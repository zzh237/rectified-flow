import torch
import random
import numpy as np

import matplotlib.pyplot as plt

from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.utils import set_seed, match_dim_with_data
from typing import Callable


class Sampler:
    def __init__( # NOTE: consider using dataclass config
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
    ):
        self.rectified_flow = rectified_flow

        # Prepare time grid
        if num_steps is not None or time_grid is not None:
            self.num_steps, self.time_grid = self._prepare_time_grid(num_steps, time_grid)
        else:
            self.num_steps = None
            self.time_grid = None

        self.callbacks = callbacks or []
        self.record_traj_period = record_traj_period

        # Initialize sampling state
        self.num_samples = num_samples
        self.x_t = None
        self.x_0 = None
        self.step_count = 0

    def _prepare_time_grid(self, num_steps, time_grid):
        if num_steps is None and time_grid is None:
            raise ValueError("At least one of num_steps or time_grid must be provided")

        if time_grid is None:
            time_grid = np.linspace(0, 1, num_steps + 1).tolist()
        else:
            if isinstance(time_grid, torch.Tensor):
                time_grid = time_grid.tolist()
            elif isinstance(time_grid, np.ndarray):
                time_grid = time_grid.tolist()
            elif not isinstance(time_grid, list):
                time_grid = list(time_grid)

            if num_steps is None:
                num_steps = len(time_grid) - 1
            else:
                assert len(time_grid) == num_steps + 1, "Time grid must have num_steps + 1 elements"

        return num_steps, time_grid

    def get_velocity(self, **model_kwargs):
        x_t, t = self.x_t, self.t
        t = match_dim_with_data(t, x_t.shape, x_t.device, x_t.dtype, expand_dim=False)
        return self.rectified_flow.get_velocity(x_t, t, **model_kwargs)

    def step(self, **model_kwargs):
        """
        Performs a single integration step.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this step method.")

    def set_next_time_point(self):
        """Advances to the next time point."""
        self.step_count += 1
        try:
            self.t = self.t_next
            self.t_next = next(self.time_iter)
        except StopIteration:
            self.t_next = None

    def stop(self):
        """Determines whether the sampling should stop."""
        return (
            self.t_next is None
            or self.t >= 1.0 - 1e-6
        )

    def record(self):
        """Records trajectories and other information."""
        if self.step_count % self.record_traj_period == 0:
            self._trajectories.append(self.x_t.detach().clone().cpu())
            self._time_points.append(self.t)

            # Callbacks can be used for logging or additional recording
            for callback in self.callbacks:
                callback(self)

    @torch.no_grad()
    def sample_loop(
        self,
        num_samples: int | None = None,
        x_0: torch.Tensor | None = None,
        seed: int | None = None,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        **model_kwargs,
    ):
        if seed is not None:
            set_seed(seed)

        if num_samples is None:
            if x_0 is not None:
                num_samples = x_0.shape[0]
            elif self.num_samples is not None:
                num_samples = self.num_samples
            else:
                raise ValueError(
                    "num_samples must be specified if x_0 is not provided."
                )
        self.num_samples = num_samples

        # Prepare initial state
        if x_0 is not None:
            self.x_t = x_0
        else:
            self.x_t = self.rectified_flow.sample_source_distribution(num_samples)
        self.x_0 = self.x_t.clone()

        # Prepare time grid, can be overridden when calling the method
        if num_steps is not None:
            self.num_steps = num_steps
        if time_grid is not None:
            self.time_grid = time_grid

        self.num_steps, self.time_grid = self._prepare_time_grid(self.num_steps, self.time_grid)
        self.step_count = 0
        self.time_iter = iter(self.time_grid)
        self.t = next(self.time_iter)
        self.t_next = next(self.time_iter)

        # Recording trajectories
        self._trajectories = [self.x_t.detach().clone().cpu()]
        self._time_points = [self.t]

        # Runs the sampling process
        while not self.stop():
            self.step(**model_kwargs)
            self.record()
            self.set_next_time_point()

        return self

    @property
    def trajectories(self) -> list[torch.Tensor]:
        """List of recorded trajectories."""
        return self._trajectories

    @property
    def time_points(self) -> list[float]:
        """List of recorded time points."""
        return self._time_points
