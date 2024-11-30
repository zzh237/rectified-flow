import torch
import warnings
from .base_sampler import Sampler
from rectified_flow.rectified_flow import RectifiedFlow
from typing import Callable


class SDESampler(Sampler):
    """
    Stochastic sampler for rectified flow with independent coupling and a Gaussian noise distribution (pi0).

    At each iteration, we decompose `X_t` into:
        X_t = alpha_t * X1_pred + beta_t * X0_pred

    If `noise_method='stable'`, we refresh noise on `X0_pred` via:
        X_t' = X_t - beta_t_noised * (X0_pred - pi0.mean()) + sqrt(beta_t^2 - (beta_t - beta_t_noised)^2) * (refresh_noise - pi0.mean())
    where `beta_t_noised` is set to:
        beta_t_noised = step_size * noise_scale(t) * beta_t(t) ** noise_decay_rate(t),
    and `beta_t_noised = min(beta_t_noised, beta_t)`.

    If `noise_method='euler'`, we use the approximation:
        sqrt(beta_t^2 - (beta_t - beta_t_noised)^2) ≈ sqrt(2 * beta_t * beta_t_noised)

    When using `noise_method='euler'` and `ode_method='euler'`, the method is equivalent to the Euler method for solving the SDE:
        dX_t = v_t(X_t) dt - e_t * (X0_pred(X_t) - pi0.mean()) dt + sqrt(2 * beta_t * e_t) * sqrt(pi0.cov()) * dW_t
    with:
        e_t = beta_t_noised / step_size = noise_scale(t) * beta_t(t) ** noise_decay_rate(t)

    Args:
        rectified_flow (RectifiedFlow): The rectified flow model.
        num_steps (int, optional): Number of steps in the time grid. Defaults to None.
        time_grid (list[float] or torch.Tensor, optional): Custom time grid. Defaults to None.
        record_traj_period (int, optional): Period to record trajectory. Defaults to 1.
        callbacks (list[Callable], optional): List of callback functions. Defaults to None.
        num_samples (int, optional): Number of samples to generate. Defaults to None.
        noise_magnitude (Callable, optional): Function to compute noise magnitude at time t. Should accept a float t and return a float. Defaults to `lambda t: 1`.
        noise_method (str, optional): Method to compute noise ('stable' or 'euler'). Defaults to 'stable'.
        ode_method (str, optional): Method to advance ODE ('straight' or 'curved'). Defaults to 'curved'.

    Attributes:
        noise_magnitude (Callable): Function to compute noise magnitude at time t.
        noise_method (str): Method to compute noise ('stable' or 'euler').
        ode_method (str): Method to advance ODE ('straight' or 'curved').

    Raises:
        ValueError: If `pi_0` is not a standard Gaussian distribution or the coupling is not independent.
    """

    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
        noise_scale: float | Callable = 1.0,
        noise_decay_rate: float | Callable = 1.0,
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
        self.noise_scale = self._process_coeffs(noise_scale)
        self.noise_decay_rate = self._process_coeffs(noise_decay_rate)
        self.noise_method = noise_method
        self.ode_method = ode_method

        if not self.rectified_flow.independent_coupling:
            warnings.warn(
                "For the sampler to be theoretically correct, the coupling must be independent. Proceed at your own risk."
            )
        if not self.rectified_flow.is_pi_0_gaussian:
            raise ValueError(
                "pi_0 should be Gaussian (torch.distributions.Normal or torch.distributions.MultivariateNormal)."
            )

    @staticmethod
    def _process_coeffs(coeff):
        if isinstance(coeff, (torch.Tensor, int, float)):
            return lambda t: coeff
        elif callable(coeff):
            return coeff
        else:
            raise TypeError("coeff should be a float, int, torch.Tensor, or callable.")

    def step(self, **model_kwargs):
        """
        Perform a single SDE sampling step.

        Args:
            **model_kwargs: Additional keyword arguments passed to the model when computing the velocity.

        Updates:
            self.x_t (Tensor): The updated sample at the next time step `t_next`.
        """
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.get_velocity(**model_kwargs)
        step_size = t_next - t

        # Solve for x_0 and x_1 given x_t and v_t
        self.rectified_flow.interp.solve(t=t, x_t=x_t, dot_x_t=v_t)
        x_0 = self.rectified_flow.interp.x_0
        x_1 = self.rectified_flow.interp.x_1
        beta_t = self.rectified_flow.interp.beta(t)

        # Part 1: Add noise

        # 1) Calculate beta_t_noised, the fraction of x_0 that will be noised
        beta_t_noised = step_size * self.noise_scale(t) * beta_t**self.noise_decay_rate(t)
        # Clip beta_t_noised to beta_t, it's not meaningful to have beta_t_noised > beta_t
        if beta_t_noised > beta_t:
            beta_t_noised = beta_t

        refresh_noise = self.rectified_flow.sample_source_distribution(self.num_samples)
        pi_0_mean = self.rectified_flow.pi_0.mean

        # 2) Remove beta_t_noised * x_0 and then add refreshed noise
        if self.noise_method.lower() == 'stable':
            noise_std = (beta_t ** 2 - (beta_t - beta_t_noised) ** 2) ** 0.5
            langevin_term = -beta_t_noised * (x_0 - pi_0_mean) + noise_std * (refresh_noise - pi_0_mean)
        # this is the taylor approximation of the stable method when beta_t_noised is small, and corresponds to Euler method for the langevin dynamics
        elif self.noise_method.lower() == 'euler':
            noise_std = (2 * beta_t * beta_t_noised) ** 0.5
            langevin_term = -beta_t_noised * (x_0 - pi_0_mean) + noise_std * (refresh_noise - pi_0_mean)

        else:
            raise ValueError(f"Unknown noise_method: {self.noise_method}")

        # print(f"t = {t:.5f}, coeff = {coeff:.5f}, noise_std = {noise_std:.5f}")

        x_t_noised = x_t + langevin_term

        # Advance time using the specified ODE method
        if self.ode_method.lower() == 'euler':
            # standard Euler method
            x_t_next = x_t_noised + step_size * v_t

        elif self.ode_method.lower() == 'curved':
            # Curved Euler method, following the underlying interpolation curve
            # a. Get x_0_noised from x_t_noised and x_1
            self.rectified_flow.interp.solve(t=t, x_t=x_t_noised, x_1=x_1)
            x_0_noised = self.rectified_flow.interp.x_0

            # b. Interpolate to get x_t_next given x_0_noised and x_1
            self.rectified_flow.interp.solve(t=t_next, x_0=x_0_noised, x_1=x_1)
            x_t_next = self.rectified_flow.interp.x_t

        else:
            raise ValueError(f"Unknown ode_method: {self.ode_method}")

        # Update the current sample
        self.x_t = x_t_next
