import torch
from .base_sampler import Sampler
from rectified_flow.rectified_flow import RectifiedFlow
from typing import Callable


class SDESampler(Sampler):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
        noise_magnitude: Callable = lambda t: 1,
        noise_method: str = 'stable',
        ode_method: str = 'curved',
    ):
        super().__init__(
            rectified_flow, 
            num_steps, 
            time_grid, 
            record_traj_period, 
            callbacks, 
            num_samples,
        )
        self.noise_magnitude = noise_magnitude
        self.noise_method = noise_method
        self.ode_method = ode_method
        if not (self.rectified_flow.is_pi0_guassian and self.rectified_flow.independent_coupling):
            raise ValueError(
                "pi_0 must be a standard Gaussian distribution, "
                "and the coupling must be independent."
            )

    def step(self, **model_kwargs):
        """Perform a single SDE sampling step."""
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.get_velocity(**model_kwargs)
        step_size = t_next - t

        # generate noise
        noise = self.rectified_flow.sample_source_distribution(self.num_samples)
        pi_0_mean = self.rectified_flow.pi_0.mean

        self.rectified_flow.interp.solve(t=t, x_t=x_t, dot_x_t=v_t)
        x_0 = self.rectified_flow.interp.x_0
        x_1 = self.rectified_flow.interp.x_1
        beta_t = self.rectified_flow.interp.beta(t)

        coeff = step_size * self.noise_magnitude(t)
        # it is not meaningful to have coeff>beta_t, clip
        if coeff > beta_t: coeff = beta_t

        # calculating noise term
        # stable method introduces slightly smaller noise, corresponding to choices in OvershootingSampler and NoiseRefreshSampler
        if self.noise_method.lower() == 'stable':
            noise_std = (beta_t ** 2 - (beta_t - coeff) ** 2) ** (0.5)
            langevin_term = -coeff * (x_0 - pi_0_mean) + noise_std * (noise - pi_0_mean)

        elif self.noise_method.lower() == 'euler':
            noise_std = (2 * beta_t * coeff) ** (0.5)
            langevin_term = -coeff * (x_0 - pi_0_mean) + noise_std * (noise - pi_0_mean)

        # print(f"t = {t:.5f}, coeff = {coeff:.5f}, noise_std = {noise_std:.5f}")

        x_t_noised = x_t + langevin_term

        # advance time, using either Euler method, or Curved Euler method
        if self.ode_method.lower() == 'straight':
            x_t_next = x_t_noised + step_size * v_t

        elif self.ode_method.lower() == 'curved':
            # get x_0_noised from x_t_noised and x_1
            self.rectified_flow.interp.solve(t=t, x_t=x_t_noised, x_1=x_1)
            x_0_noised = self.rectified_flow.interp.x_0

            # interp to get x_t_next given x_0_noised and x_1
            self.rectified_flow.interp.solve(t=t_next, x_0=x_0_noised, x_1=x_1)
            x_t_next = self.rectified_flow.interp.x_t

        self.x_t = x_t_next
