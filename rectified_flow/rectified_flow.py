import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
import warnings

import torch.distributions as dist
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .flow_components.interpolation_solver import AffineInterp
from .flow_components.train_time_sampler import TrainTimeSampler
from .flow_components.train_time_weight import TrainTimeWeight
from .flow_components.loss_function import RectifiedFlowLossFunction
from .utils import match_dim_with_data


class RectifiedFlow:
    def __init__(
        self,
        data_shape: tuple,
        model: nn.Module,
        interp: AffineInterp | str = "straight",
        source_distribution: torch.distributions.Distribution | str | Callable = "normal",
        is_independent_coupling: bool = True,
        train_time_distribution: TrainTimeSampler | str = "uniform",
        train_time_weight: TrainTimeWeight | str = "uniform",
        criterion: RectifiedFlowLossFunction | str = "mse",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.data_shape = data_shape
        self.model = model 

        self.interp: AffineInterp = (
            interp if isinstance(interp, AffineInterp) else AffineInterp(interp)
        )
        self.train_time_sampler: TrainTimeSampler = (
            train_time_distribution
            if isinstance(train_time_distribution, TrainTimeSampler)
            else TrainTimeSampler(train_time_distribution)
        )
        self.train_time_weight: TrainTimeWeight = (
            train_time_weight
            if isinstance(train_time_weight, TrainTimeWeight)
            else TrainTimeWeight(train_time_weight)
        )
        self.criterion: RectifiedFlowLossFunction = (
            criterion if isinstance(criterion, RectifiedFlowLossFunction) else RectifiedFlowLossFunction(criterion)
        )
        
        self.pi_0 = source_distribution
        if self.pi_0 == "normal": 
            self.pi_0 = dist.Normal(
                torch.tensor(0, device=device, dtype=dtype),
                torch.tensor(1, device=device, dtype=dtype)
            ).expand(data_shape)
        else:
            if isinstance(self.pi_0, dist.Distribution):
                if self.pi_0.mean.device != device or self.pi_0.stddev.device != device:
                    warnings.warn("Source distribution device does not match the model device.")
                if self.pi_0.mean.dtype != dtype or self.pi_0.stddev.dtype != dtype:
                    warnings.warn("Source distribution dtype does not match the model dtype.")

        self.independent_coupling = is_independent_coupling

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = torch.dtype(dtype) if isinstance(dtype, str) else dtype

    def sample_train_time(self, batch_size: int):
        return self.train_time_sampler(batch_size, device=self.device, dtype=self.dtype)

    def sample_source_distribution(self, batch_size: int):
        if isinstance(self.pi_0, dist.Distribution):
            return self.pi_0.sample((batch_size,)).to(self.device, self.dtype)
        elif callable(self.pi_0):
            return self.pi_0(batch_size).to(self.device, self.dtype)
        else:
            raise ValueError("Source distribution must be a torch.distributions.Distribution or a callable.")

    def get_interpolation(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Compute the interpolated values X_t and dot_Xt at time t

        Args:
            x_0 (torch.Tensor): X_0, shape (B, D) or (B, D1, D2, ..., Dn)
            x_1 (torch.Tensor): X_1, shape (B, D) or (B, D1, D2, ..., Dn)
            t (torch.Tensor): Time tensor, shape (B,), in [0, 1], no need for match_dim_with_data

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: x_t, dot_x_t, both of shape (B, D) or (B, D1, D2, ..., Dn)
        """
        assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape."
        assert x_0.shape[0] == x_1.shape[0], "Batch size of X_0 and X_1 must match."
        x_t, dot_x_t = self.interp.forward(x_0, x_1, t, detach=True)
        return x_t, dot_x_t

    def get_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ):
        """
        Compute the velocity of the flow at (X_t, t)
        Decouples velocity computation from the model forward pass, handle t transformation, etc.

        Args:
            x_t (torch.Tensor): X_t, shape (B, D) or (B, D1, D2, ..., Dn)
            t (torch.Tensor): Time tensor, shape (B,), in [0, 1], no need for match_dim_with_data

        Returns:
            torch.Tensor: Velocity tensor, same shape as X
        """
        assert x_t.shape[0] == t.shape[0] and t.ndim == 1, "Batch size of x_t and t must match."
        velocity = self.model(x_t, t, **kwargs)
        return velocity
    
    def get_loss(
        self,
        x_0: torch.Tensor | None,
        x_1: torch.Tensor,
        t: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Compute the loss of the flow model(X_t, t)

        Args:
            x_0 (torch.Tensor): X_0, shape (B, D) or (B, D1, D2, ..., Dn), can be None
                                Must be provided to avoid ambiguity in passing arguments
            x_1 (torch.Tensor): X_1, shape (B, D) or (B, D1, D2, ..., Dn)
            t (torch.Tensor): Time tensor, shape (B,), in [0, 1], optional
            **kwargs: Additional keyword arguments for the model input

        Returns:
            torch.Tensor: Loss tensor, scalar
        """
        t = self.sample_train_time(x_1.shape[0]) if t is None else t
        x_0 = self.sample_source_distribution(x_1.shape[0]) if x_0 is None else x_0

        x_t, dot_x_t = self.get_interpolation(x_0, x_1, t)
        v_t = self.get_velocity(x_t, t, **kwargs)
        time_weights = self.train_time_weight(t)

        return self.criterion(
            v_t=v_t,
            dot_x_t=dot_x_t,
            x_t=x_t,
            t=t,
            time_weights=time_weights,
        )
    
    def get_score_function(self, x_t, t, **kwargs):
        v_t = self.get_velocity(x_t, t, **kwargs)
        return self.get_score_function_from_velocity(x_t, v_t, t)

    def get_score_function_from_velocity(self, x_t, v_t, t):
        """
        Compute the score function of the flow at (X_t, t) from the velocity.
        
        pi_0 (source distribution) must a Gaussian distribution.
        If pi_0 is Normal(0, I), we calculate based on Dlogpt(X_t) = -E[X_0|X_t]/bt. 
        Otherwise, pi_0.score_function must be provided.
        """
        if not self.independent_coupling or not self.is_pi_0_gaussian:
            warnings.warning('The formula is theoretically correct only for independent couplings and Gaussian pi0, use at your own risk')

        self.interp.solve(t=t, x_t=x_t, dot_x_t=v_t)
        dlogp = self.get_score_function_of_pi_0(self.interp.x_0) / self.interp.b_t
        return dlogp
    
    def get_score_function_of_pi_0(self, x_0):
        """
        Compute Dlogp_0(X_0), the score function  of the source distribution pi_0 at X_0.
        """
        if self.is_pi_0_standard_gaussian:
            return -x_0    
        elif isinstance(self.pi_0, torch.distributions.Normal):            
            return -(x_0 - self.pi_0.mean.to(self.device, self.dtype)) / self.pi_0.variance.to(self.device, self.dtype)
        elif isinstance(self.pi_0, torch.distributions.MultivariateNormal):
            return -(x_0 - self.pi_0.mean.to(self.device, self.dtype)) @ self.pi_0.precision_matrix.to(self.device, self.dtype)
        else:   
            try: 
                return self.pi_0.score_function(x_0)                        
            except:
                raise ValueError('pi_0 is not a standard Gaussian distribution and must provide a score function.')  

    def get_sde_params_by_sigma(self, v_t, x_t, t, sigma):
        # SDE coeffs for dX_t = v_t(X_t) + sigma_t^2*Dlogp(X_t) + sqrt(2)*sigma_t*dWt
        self.assert_canonical()
        sigma_t = sigma(t)
        self.interp.solve(t=t, x_t=x_t, dot_x_t=v_t)
        dlogp = - self.interp.x0 / self.interp.bt
        v_t_sde = v_t + sigma_t**2 * dlogp
        return v_t_sde, sigma_t * 2**0.5
    
    def get_stable_sde_params(self, v_t, x_t, t, e):
        # From SDE coeffs for dX = v_t(Xt) -sigma_t^2*E[X0|Xt]/bt + sqrt(2)*sigma_t*dWt,
        # let et^2 = sigmat^2/bt, we have sigmat = sqrt(bt) * et, we have:
        # dX = v_t(Xt) - et^2*E[X0|Xt]+ sqrt(2*bt) * et *dWt
        self.assert_canonical()
        self.interp.solve(t=t, x_t=x_t, dot_x_t=v_t)
        et = e(self.match_dim_with_data(t, x_t.shape, device=x_t.device, dtype=x_t.dtype))
        x0_pred  = - self.interp.x0/self.interp.bt
        v_t_sde = v_t - x0_pred * et**2
        sigma_t = et * self.interp.bt**0.5 * (2**0.5)
        # at, bt, dot_at, dot_bt = self.interp.get_coeffs(t)
        # v_t_sde =v_t * (1+et) - et * dot_at / at * xt
        # sigma_t_sde = (2 * (1-at) * dot_at/(at) * et)**(0.5)
        return v_t_sde, sigma_t
    
    def match_dim_with_data(
        self,
        t: torch.Tensor | float | List[float],
        x_shape: tuple,
        expand_dim: bool = True,
    ):
        return match_dim_with_data(t, x_shape, device=self.device, dtype=self.dtype, expand_dim=expand_dim)

    @property
    def is_pi_0_gaussian(self):
        """Check if pi_0 is a Gaussian distribution."""
        return isinstance(self.pi_0, dist.Normal) or isinstance(self.pi_0, dist.MultivariateNormal)

    @property
    def is_pi_0_zero_mean_gaussian(self):
        """Check if pi_0 is a zero-mean Gaussian distribution."""
        if callable(self.pi_0): return True # NOTE: fix this

        is_multivariate_normal = (
            isinstance(self.pi_0, dist.MultivariateNormal) and 
            torch.allclose(self.pi_0.mean, torch.zeros_like(self.pi_0.mean))
        )
        is_normal = (
            isinstance(self.pi_0, dist.Normal) and 
            torch.allclose(self.pi_0.loc, torch.zeros_like(self.pi_0.loc))
        )
        return is_multivariate_normal or is_normal
    
    @property
    def is_pi_0_standard_gaussian(self):
        """Check if pi_0 is a standard Gaussian distribution."""
        is_multivariate_normal = (
            isinstance(self.pi_0, dist.MultivariateNormal) and
            torch.allclose(self.pi_0.mean, torch.zeros_like(self.pi_0.mean)) and
            torch.allclose(
                self.pi_0.covariance_matrix,
                torch.eye(self.pi_0.mean.size(0), device=self.pi_0.mean.device)
            )
        )
        is_normal = (
            isinstance(self.pi_0, dist.Normal) and
            torch.allclose(self.pi_0.mean, torch.zeros_like(self.pi_0.mean)) and
            torch.allclose(self.pi_0.variance, torch.ones_like(self.pi_0.variance))
        )
        return is_multivariate_normal or is_normal
    
    @property
    def is_independent_coupling(self):
        """Check if rectified flow is a independent coupling."""
        return self.independent_coupling

    @property
    def is_canonical(self):
        """Check if the rectified flow is in canonical form."""
        return self.is_pi_0_standard_gaussian and self.is_independent_coupling
