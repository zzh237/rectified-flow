import torch
import random
import numpy as np

import matplotlib.pyplot as plt

from rectified_flow.rectified_flow import RectifiedFlow, match_dim_with_data
from utils.utils import set_seed

class Sampler:
    ODE_SAMPLING_STEP_LIMIT = 1000

    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int = 1,
        seed: int = 0,
        callbacks: list[callable] | None = None,
        num_samples: int | None = None,
    ):
        self.seed = seed
        set_seed(seed)

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
        self.X_t = None
        self.X_0 = None
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
        X_t, t = self.X_t, self.t
        t = match_dim_with_data(t, X_t.shape, X_t.device, X_t.dtype, expand_dim=False)
        return self.rectified_flow.get_velocity(X_t, t, **model_kwargs)

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
            or self.step_count >= self.ODE_SAMPLING_STEP_LIMIT
            or self.t >= 1.0 - 1e-6
        )

    def record(self):
        """Records trajectories and other information."""
        if self.step_count % self.record_traj_period == 0:
            self._trajectories.append(self.X_t.detach().clone().cpu())
            self._time_points.append(self.t)

            # Callbacks can be used for logging or additional recording
            for callback in self.callbacks:
                callback(self)

    @torch.inference_mode()
    def sample_loop(
        self,
        num_samples: int | None = None,
        X_0: torch.Tensor | None = None,
        seed: int | None = None,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        **model_kwargs,
    ):
        if seed is not None:
            set_seed(seed)

        if num_samples is None:
            if X_0 is not None:
                num_samples = X_0.shape[0]
            elif self.num_samples is not None:
                num_samples = self.num_samples
            else:
                raise ValueError(
                    "num_samples must be specified if X_0 is not provided."
                )
        self.num_samples = num_samples

        # Prepare initial state
        if X_0 is not None:
            self.X_t = X_0
        else:
            self.X_t = self.rectified_flow.sample_source_distribution(num_samples)
        self.X_0 = self.X_t.clone()

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
        self._trajectories = [self.X_t.clone().cpu()]
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


class EulerSampler(Sampler):
    def step(self, **model_kwargs):
        t, t_next, X_t = self.t, self.t_next, self.X_t
        v_t = self.get_velocity(**model_kwargs)
        self.X_t = X_t + (t_next - t) * v_t


class CurvedEulerSampler(Sampler):
    def step(self, **model_kwargs):
        t, t_next, X_t = self.t, self.t_next, self.X_t
        v_t = self.get_velocity(**model_kwargs)

        self.rectified_flow.interp.solve(t, xt=X_t, dot_xt=v_t)
        X_1_pred = self.rectified_flow.interp.x1
        X_0_pred = self.rectified_flow.interp.x0

        # interplate to find x_{t_next}
        self.rectified_flow.interp.solve(t_next, x0=X_0_pred, x1=X_1_pred)
        self.X_t = self.rectified_flow.interp.xt


class NoiseRefreshSampler(Sampler):
    def __init__(self, noise_replacement_rate=lambda t: 0.5, **kwargs): 
        super().__init__(**kwargs)  
        self.noise_replacement_rate = noise_replacement_rate
        assert (self.rectified_flow.independent_coupling and self.rectified_flow.is_pi0_zero_mean_gaussian), \
            'pi0 must be a zero mean gaussian and must use indepdent coupling'

    def step(self, **model_kwargs):
        """Perform a single step of the sampling process."""
        t, t_next, X_t = self.t, self.t_next, self.X_t
        v_t = self.get_velocity(**model_kwargs)

        # Given xt and dot_xt = vt, find the corresponding endpoints x0 and x1
        self.rectified_flow.interp.solve(t, xt=X_t, dot_xt=v_t)
        X_1_pred = self.rectified_flow.interp.x1
        X_0_pred = self.rectified_flow.interp.x0

        # Randomize x0_pred by replacing part of it with new noise
        noise = self.rectified_flow.sample_source_distribution(self.num_samples)
        noise_replacement_factor = self.noise_replacement_rate(t)
        X_0_pred_refreshed = (
            (1 - noise_replacement_factor**2)**0.5 * X_0_pred +
            noise * noise_replacement_factor
        )

        # Interpolate to find xt at t_next
        self.rectified_flow.interp.solve(t_next, x0=X_0_pred_refreshed, x1=X_1_pred)
        self.X_t = self.rectified_flow.interp.xt


class OverShootingSampler(Sampler):
    def __init__(self, c=1.0, overshooting_method='t + dt * (1 - t)', **kwargs):
        super().__init__(**kwargs)
        self.c = c

        # Define overshooting method
        if callable(overshooting_method):
            self.overshooting = overshooting_method
        elif isinstance(overshooting_method, str):
            try:
                self.overshooting = eval(f"lambda t, dt: {overshooting_method}")
            except SyntaxError:
                raise ValueError(f"Invalid overshooting method: {overshooting_method}")
        else:
            raise ValueError("Invalid overshooting method provided. Must be a string or callable.")

        # Ensure rf meets required conditions
        if not (self.rectified_flow.is_pi0_zero_mean_gaussian() and self.rectified_flow.independent_coupling):
            raise ValueError(
                "pi0 must be a zero-mean Gaussian distribution, and the coupling must be independent."
            )

    def step(self, **model_kwargs):
        """Perform a single overshooting step."""
        t, t_next, X_t = self.t, self.t_next, self.X_t
        v_t = self.get_velocity(**model_kwargs)
        alpha = self.rectified_flow.interp.alpha
        beta = self.rectified_flow.interp.beta

        # Calculate overshoot time and enforce constraints
        t_overshoot = min(self.overshooting(t_next, (t_next - t) * self.c), 1.0)
        if t_overshoot < t_next:
            raise ValueError("t_overshoot cannot be smaller than t_next.")

        # Advance to t_overshoot using ODE
        X_t_overshoot = X_t + (t_overshoot - t) * v_t

        # Apply noise to step back to t_next
        at = alpha(t_next) / alpha(t_overshoot)
        bt = (beta(t_next)**2 - (at * beta(t_overshoot))**2)**0.5
        noise = self.rectified_flow.sample_source_distribution(self.num_samples)
        self.X_t = X_t_overshoot * at + noise * bt
        

class SDESampler(Sampler):
    def __init__(self, e=lambda t: torch.ones_like(t), **kwargs):
        super().__init__(**kwargs)
        self.e = e

        if not (self.rectified_flow.is_pi0_standard_gaussian() and self.rectified_flow.independent_coupling):
            raise ValueError(
                "pi0 must be a standard Gaussian distribution, "
                "and the coupling must be independent."
            )

    def step(self, **model_kwargs):
        """Perform a single SDE sampling step."""
        t, t_next, X_t = self.t, self.t_next, self.X_t
        v_t = self.get_velocity(**model_kwargs)

        # Prepare time tensor and ensure it is within bounds
        t_ones = t * torch.ones((X_t.shape[0],), device=X_t.device)
        t_ones = match_dim_with_data(t_ones, X_t.shape, X_t.device, X_t.dtype, expand_dim=True)
        t_eps = 1e-12
        t_ones = torch.clamp(t_ones, t_eps, 1 - t_eps)
        step_size = t_next - t

        # Calculate alpha, beta, and their gradients
        e_t = self.e(t_ones)
        a_t, b_t, dot_a_t, dot_b_t = self.rectified_flow.interp.get_coeffs(t_ones)

        # Adjust velocity and calculate noise scale
        # print(f"e_t: {e_t.shape}, a_t: {a_t.shape}, b_t: {b_t.shape}, dot_a_t: {dot_a_t.shape}, dot_b_t: {dot_b_t.shape}")
        v_adj_t = (1 + e_t) * v_t - e_t * dot_a_t / a_t * X_t
        sigma_t = torch.sqrt(2 * (b_t**2 * dot_a_t / a_t - dot_b_t * b_t) * e_t)

        # Predict x1 and update xt with noise
        X_1_pred = X_t + (1 - t) * v_t
        noise = self.rectified_flow.sample_source_distribution(self.num_samples)
        self.X_t = X_t + step_size * v_adj_t + sigma_t * step_size**0.5 * noise