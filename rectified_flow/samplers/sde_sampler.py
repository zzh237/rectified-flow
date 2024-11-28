import torch
from .base_sampler import Sampler
from rectified_flow.utils import match_dim_with_data


class SDESampler(Sampler):
    def __init__(self, diffusion_coefficient=lambda t: 1, **kwargs):
        super().__init__(**kwargs)
        self.diffusion_coefficient = diffusion_coefficient

        if not (self.rectified_flow.is_pi0_standard_gaussian and self.rectified_flow.independent_coupling):
            raise ValueError(
                "pi0 must be a standard Gaussian distribution, "
                "and the coupling must be independent."
            )

    def step(self, **model_kwargs):
        """Perform a single SDE sampling step."""
        t, t_next, X_t = self.t, self.t_next, self.X_t
        v_t = self.get_velocity(**model_kwargs)
        step_size = t_next - t

        # get dlogp_0, by Tweddie's formula, dlogp_t(x) = E[dlogp_0(X0)|Xt]/beta_t
        if self.rectified_flow.is_pi0_standard_gaussian:
            dlogp0 = lambda x: -x
        else:
            try:
                dlogp0 = self.pi_0.score_function
            except:
                print("we must define a score_function attribute for pi_0")

        # generate noise
        noise = self.rectified_flow.sample_source_distribution(self.num_samples)

        self.rectified_flow.interp.solve(t=t, xt=X_t, dot_xt=v_t)
        x0 = self.rectified_flow.interp.x0
        beta_t = self.rectified_flow.interp.bt
        coeff = self.diffusion_coefficient(t)

        langevin_term = (coeff * step_size * dlogp0(x0)
                         + (2 * step_size * beta_t * coeff) ** (0.5) * noise)

        X_t_next = X_t + step_size * v_t + langevin_term
        self.X_t = X_t_next
