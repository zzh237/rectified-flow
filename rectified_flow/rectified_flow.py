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

def match_time_dim_with_data(
    t: torch.Tensor | float | List[float],
    X_shape: tuple,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    """
    Prepares the time tensor by reshaping it to match the dimensions of X.

    Args:
        t (Union[torch.Tensor, float, List[float]]): Time tensor, which can be:
            - A scalar (float or 0-dimensional torch.Tensor)
            - A list of floats with length equal to the batch size
            - A torch.Tensor of shape (B,)
        X_shape (tuple): Shape of the tensor X, e.g., X.shape

    Returns:
        torch.Tensor: Reshaped time tensor, ready for broadcasting with X.
    """
    B = X_shape[0]  # Batch size

    if isinstance(t, float):
        t = torch.full((B,), t, device=device, dtype=dtype)
    elif isinstance(t, list):
        if len(t) != B:
            raise ValueError(f"Length of t list ({len(t)}) does not match batch size ({B}).")
        t = torch.tensor(t, device=device, dtype=dtype)
    elif isinstance(t, torch.Tensor):
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.repeat(B)
        elif t.shape[0] != B:
            raise ValueError(f"Batch size of t ({t.shape[0]}) does not match X ({B}).")
    else:
        raise TypeError(f"t must be a torch.Tensor, float, or a list of floats, but got {type(t)}.")

    # Reshape t to have singleton dimensions matching X_shape after the batch dimension
    expanded_dims = [1] * (len(X_shape) - 1)
    t = t.view(B, *expanded_dims)
    return t

class AffineInterpSolver:
    '''
    This is symbolic solver for equation: xt = at*x1+bt*x0, dot_xt = dot_at*x1+dot_bt*x0
    Given the known variables, and keep the unknowns as None; the solver will fill the unknowns automatically.
    '''
    def __init__(self):
        x0s, x1s, xts, dot_xts, ats, bts, dot_ats, dot_bts = sympy.symbols('r.x0 r.x1 r.xt r.dot_xt r.at r.bt r.dot_at r.dot_bt')
        eq1 = sympy.Eq(xts, ats * x1s + bts * x0s)
        eq2 = sympy.Eq(dot_xts, dot_ats * x1s + dot_bts * x0s)
        # create symbolic solver for all pairs of unknown variables
        self.symbolic_solvers = {}
        variables = [x0s, x1s, xts, dot_xts]
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                unknown1, unknown2 = variables[i], variables[j]
                solution = sympy.solve((eq1, eq2), (unknown1, unknown2))
                func = eval(f'lambda r: '
                            f'({str(solution[unknown1])}, {str(solution[unknown2])})')
                obj_name, attr_name1 = str(unknown1).split('.')
                obj_name, attr_name2 = str(unknown2).split('.')
                self.symbolic_solvers[(attr_name1, attr_name2)] = func

    def solve(self, results):
        r = results
        known_vars = {k: v for k, v in zip(['x0', 'x1', 'xt', 'dot_xt'], [r.x0, r.x1, r.xt, r.dot_xt]) if v is not None}
        unknown_vars = {k: v for k, v in zip(['x0', 'x1', 'xt', 'dot_xt'], [r.x0, r.x1, r.xt, r.dot_xt]) if v is None}
        unknown_keys = tuple(unknown_vars.keys())  # No sorting, preserving order

        if len(unknown_keys) > 2:
            warnings.warn("At least two variables in (x0, x1, xt, dot_xt) must be specified to use the solver.", UserWarning)
        elif len(unknown_keys) == 0:
            return r
        elif len(unknown_keys) == 1:
            first_known_key = next(iter(known_vars.keys()))
            unknown_keys.append(first_known_key)

        func = self.symbolic_solvers.get(unknown_keys)

        solved_value1, solved_value2 = func(r)

        unknown1 = unknown_keys[0]; setattr(r, unknown1, solved_value1)
        unknown2 = unknown_keys[1]; setattr(r, unknown2, solved_value2)

        return r


class AffineInterp(nn.Module):
    def __init__(self, alpha=None, beta=None, dot_alpha=None, dot_beta = None, name = 'affine'):
        super().__init__()

        self._process_defaults(alpha, beta, dot_alpha, dot_beta, name)

        self.solver = AffineInterpSolver()
        self.at = None; self.bt = None; self.dot_at = None; self.dot_bt = None
        self.x0 = None; self.x1 = None; self.xt = None; self.dot_xt = None

    def _process_defaults(self, alpha, beta, dot_alpha, dot_beta, name):
        if (isinstance(alpha, str) and (alpha.lower() in ['straight', 'lerp'])) or (alpha is None and beta is None):
            # Special case for 'straight' interpolation
            alpha = lambda t: t
            beta = lambda t: 1 - t
            dot_alpha = lambda t: 1.0
            dot_beta = lambda t: -1.0
            name = 'straight'

        elif isinstance(alpha, str) and alpha.lower() in ['harmonic', 'cos', 'sin', 'slerp', 'spherical']:
            # Special case of spherical interpolation
            alpha = lambda t: torch.sin(t * torch.pi/2.0)
            beta = lambda t: torch.cos(t * torch.pi/2.0)
            dot_alpha = lambda t: torch.cos(t * torch.pi/2.0) * torch.pi/2.0
            dot_beta = lambda t: -torch.sin(t * torch.pi/2.0) * torch.pi/2.0
            name = 'spherical'

        elif isinstance(alpha, str) and alpha.lower() in ['ddim', 'ddpm']:
            # DDIM/DDPM scheme; see Eq 7 in https://arxiv.org/pdf/2209.03003
            a = 19.9; b = 0.1
            alpha = lambda t: torch.exp(-a*(1-t)**2/4.0 - b*(1-t)/2.0)
            beta = lambda t: (1-alpha(t)**2)**(0.5)
            name = 'DDIM'

        self.alpha = lambda t: alpha(self.ensure_tensor(t))
        self.beta = lambda t: beta(self.ensure_tensor(t))
        if dot_alpha is not None: self.dot_alpha = lambda t: dot_alpha(self.ensure_tensor(t))
        if dot_beta is not None: self.dot_beta = lambda t: dot_beta(self.ensure_tensor(t))
        self.name = name

    @staticmethod
    def ensure_tensor(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x

    @staticmethod
    def value_and_grad(f, input, detach=True):
        x = input.clone(); x.requires_grad_(True)
        with torch.enable_grad():
            value = f(x)
            grad, = torch.autograd.grad(value.sum(), x, create_graph=(detach==False))
        if detach: value = value.detach(); grad = grad.detach()
        return value, grad

    def get_coeffs(self, t, detach=True):
        if self.dot_alpha is None:
            at, dot_at = self.value_and_grad(self.alpha, t, detach=detach)
        else:
            at = self.alpha(t); dot_at = self.dot_alpha(t)
        if self.dot_beta is None:
            bt, dot_bt = self.value_and_grad(self.beta, t, detach=detach)
        else:
            bt = self.beta(t); dot_bt = self.dot_beta(t)
        self.at = at; self.bt = bt; self.dot_at = dot_at; self.dot_bt = dot_bt
        return at, bt, dot_at, dot_bt

    def forward(self, X0, X1, t, detach=True):
        t = self.match_time_dim(t, X1)
        a_t, b_t, dot_a_t, dot_b_t = self.get_coeffs(t, detach=detach)
        Xt = a_t * X1 + b_t * X0
        dot_Xt = dot_a_t * X1 + dot_b_t * X0
        return Xt, dot_Xt

    # Solve equation: xt = at*x1+bt*x0, dot_xt = dot_at*x1+dot_bt*x0
    # Set any of the known variables, and keep the unknowns as None; the solver will fill the unknowns.
    # Example: interp.solve(t, xt=xt, dot_xt=dot_xt); print(interp.x1.shape), print(interp.x0.shape)
    def solve(self, t=None, x0=None, x1=None, xt=None, dot_xt=None):
        if t is None: raise ValueError("t must be provided")
        self.x0 = x0; self.x1 = x1; self.xt = xt; self.dot_xt = dot_xt
        x_not_none = next((v for v in [x0, x1, xt, dot_xt] if v is not None), None)
        t = self.match_time_dim(t, x_not_none)
        at, bt, dot_at, dot_bt = self.get_coeffs(t)
        self.solver.solve(self)

    # mannually coded solution for solving (x0, dot_xt) given (xt, x1)
    def x0_dot_xt_given_xt_and_x1(self, xt, t, x1):
        t = self.match_time_dim(t, xt)
        a_t, b_t, dot_a_t, dot_b_t = self.get_coeffs(t)
        x0 = (xt - a_t * x1)/b_t
        dot_xt = dot_a_t * x1 + dot_b_t * x0
        return x0, dot_xt

    # mannually coded solution for solving (x0, x1) given (xt, dot_xt)
    def x0_x1_given_xt_and_dot_xt(self, xt, dot_xt, t):
        t = self.match_time_dim(t, xt)
        a_t, b_t, dot_a_t, dot_b_t = self.get_coeffs(t)
        x0 = (dot_a_t * xt - a_t* dot_xt)/(dot_a_t * b_t - a_t * dot_b_t)
        x1 = (dot_b_t * xt - b_t* dot_xt)/(dot_b_t * a_t - b_t * dot_a_t)
        return x0, x1

def test_affine_interp():
    interp = AffineInterp('straight')
    t = torch.rand(1)
    x0 = None#torch.tensor(0.0)
    x1 = None#torch.tensor(1.0)
    xt = torch.rand(10,2)
    dot_xt = torch.randn(10,2)

    interp.solve(t,  xt=xt, dot_xt=dot_xt)
    print(interp.x0.shape)
    print(interp.x1.shape)

class TimeDistribution:
    def __init__(
        self,
        dist: str = "uniform",
    ):
        self.dist = dist

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self.dist == "uniform":
            t = torch.rand((batch_size,), device=device, dtype=dtype)
        else: # NOTE: will implement different time distributions
            raise NotImplementedError(f"Time distribution '{self.dist}' is not implemented.")
        
        return t

class TimeWeights:
    def __init__(
        self,
        weight: str = "uniform",
    ):
        self.weight = weight

    def __call__(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.weight == "uniform":
            wts = torch.ones_like(t)
        else:
            raise NotImplementedError(f"Time weight '{self.weight}' is not implemented.")
        
        return wts

class RFLossFunction:
    def __init__(
        self,
        loss_type: str = "mse",
    ):
        self.loss_type = loss_type
    
    def __call__(
        self,
        v_t: torch.Tensor,
        dot_Xt: torch.Tensor,
        X_t: torch.Tensor,
        t: torch.Tensor,
        time_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == "mse":
            loss = torch.mean(time_weights * (v_t - dot_Xt)**2)
        else:
            raise NotImplementedError(f"Loss function '{self.loss_type}' is not implemented.")
        
        return loss

class RectifiedFlow:
    def __init__(
        self,
        flow_model: nn.Module = None,
        interp_func: AffineInterp | str = "straight",
        time_dist: TimeDistribution | str = "uniform",
        time_weight: TimeWeights | str = "uniform",
        criterion: RFLossFunction | str = "mse",
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ):
        self.flow_model = flow_model 

        self.get_interpolation = interp_func if isinstance(interp_func, AffineInterp) else AffineInterp(interp_func)
        self.sample_time = time_dist if isinstance(time_dist, TimeDistribution) else TimeDistribution(time_dist)
        self.time_weight = time_weight if isinstance(time_weight, TimeWeights) else TimeWeights(time_weight)
        self.criterion = criterion if isinstance(criterion, RFLossFunction) else RFLossFunction(criterion)
        
        self.device = device
        self.dtype = dtype
        self.seed = seed

    def get_velocity(
        self,
        X: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ):
        """
        Compute the velocity of the flow at (X_t, t)
        Decouples velocity computation from the model forward pass, handle t transformation, etc.

        Args:
            X (torch.Tensor): X_t, shape (B, D) or (B, D1, D2, ..., Dn)
            t (torch.Tensor): Time tensor, shape (B,), in [0, 1], no need for match_time_dim_with_data

        Returns:
            torch.Tensor: Velocity tensor, same shape as X
        """
        assert X.shape[0] == t.shape[0] and t.ndim == 1, "Batch size of X and t must match."
        # NOTE: May do t / velocity transformation, e.g. t = 1 - t, velocity = -velocity
        velocity = self.flow_model(X, t, **kwargs)
        return velocity

    def get_loss(self, X0, X1, **kwargs):
        t = self.draw_time(X0.shape[0])
        Xt, dot_Xt = self.interp(X0, X1, t)
        vt = self.model(Xt, t, **kwargs)
        wts = self.time_weight(t)
        loss = self.criterion(vt, dot_Xt, Xt, t, wts)
        return loss
    
    def get_loss(
        self,
        X_0: torch.Tensor,
        X_1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ):
        t = self.sample_time(X_0.shape[0])
        X_t, dot_Xt = self.get_interpolation(X_0, X_1, t)
        v_t = self.get_velocity(X_t, t, **kwargs)
        wts = self.time_weight(t)
        loss = self.criterion(v_t, dot_Xt, X_t, t, wts)
        return loss

    # D_0 must ~ Normal(0,I), Dlogpt(Xt) = -E[X0 | Xt] / bt
    def get_score_function(self, Xt, vt, t):
        self.assert_canonical()
        self.interp.solve(t=t, xt=Xt, dot_xt=vt)
        dlogp = - self.interp.x0/self.interp.bt
        return dlogp

    def score_function(self, Xt, t, *args, **kwargs):
        t = self.match_time_dim(t, Xt)
        vt = self.model(Xt, t, *args, **kwargs)
        return self.get_score_function(Xt, vt, t)

    # SDE coeffs for dX = vt(Xt)+ .5*sigma_t^2*Dlogp(Xt) + sigma_t*dWt
    def get_sde_params_by_sigma(self, vt, xt, t, sigma):
        self.assert_canonical()
        sigma_t = sigma(t)
        self.interp.solve(t=t, xt=xt, dot_xt=vt)
        dlogp = - self.interp.x0/self.interp.bt
        vt_sde = vt + 0.5* sigma_t**2 * dlogp
        return vt_sde, sigma_t

    # From SDE coeffs for dX = vt(Xt) - .5*sigma_t^2*E[X0|Xt]/bt + sigma_t*dWt,
    # let et^2 = sigmat^2/bt, we have sigmat = sqrt(bt) * et, we have:
    # dX = vt(Xt) - .5*et^2*E[X0|Xt]+ sqrt(bt) * et *dWt
    def get_stable_sde_params(self, vt, xt, t, e):

        self.assert_canonical()
        self.interp.solve(t=t, xt=xt, dot_xt=vt)
        et = e(self.match_time_dim(t,xt))
        x0_pred  = - self.interp.x0/self.interp.bt
        vt_sde = vt - x0_pred * et**2 * 0.5
        sigma_t = et * self.interp.bt**0.5

        #at, bt, dot_at, dot_bt = self.interp.get_coeffs(t)
        #vt_sde =vt * (1+et) - et * dot_at / at * xt
        #sigma_t_sde = (2 * (1-at) * dot_at/(at) * et)**(0.5)
        return vt_sde, sigma_t


    def train(self, num_iterations=100, num_epochs=None, batch_size=64, D0=None, D1=None, optimizer=None, shuffle=True):
        # Will be deprecated!!!
        if optimizer is None: optimizer = self.optimizer
        if num_iterations is None: num_iterations = self.num_iterations
        if num_epochs is not None: num_epochs = self.num_epochs
        self.loss_curve = []

        # Set up dataloader
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        self.dataloader = dataloader

        # Calculate the total number of batches to process
        total_batches = num_iterations
        if num_epochs is not None:
            total_batches = num_epochs * len(dataloader)

        batch_count = 0
        while batch_count < total_batches:
            for batch in dataloader: # batch can be either (X0,X1) or (X0,X1,labels)

                if batch_count >= total_batches: break
                optimizer.zero_grad()

                # loss & backprop
                loss = self.get_loss(*batch)
                loss.backward()
                optimizer.step()

                # Track loss
                self.loss_curve.append(loss.item())
                batch_count += 1

    def is_pi0_zero_mean_gaussian(self):
        # Will be deprecated!!!
        # Check if pi0 is a zero-mean Gaussian distribution
        case1 = isinstance(self.pi0, dist.MultivariateNormal)  and torch.allclose(self.pi0.mean, torch.zeros_like(self.pi0.mean))
        case2 = isinstance(self.pi0, dist.Normal)  and torch.allclose(self.pi0.mean, torch.zeros_like(self.pi0.loc))
        return case1 or case2

    def is_pi0_standard_gaussian(self):
        # Will be deprecated!!!
        # Check if pi0 is Standard Gaussian
        case1 = isinstance(self.pi0, dist.MultivariateNormal)  \
                and torch.allclose(self.pi0.mean, self.pi0.mean*0) \
                and torch.allclose(self.pi0.covariance_matrix, torch.eye(self.pi0.mean.size(0), device=self.pi0.mean.device))
        case2 = isinstance(self.pi0, dist.Normal)  \
                and torch.allclose(self.pi0.mean, self.pi0.mean*0) \
                and torch.allclose(self.pi0.variance, self.pi0.variance*0 + 1)
        return case1 or case2

    def assert_if_pi0_is_standard_gaussian(self):
        # Will be deprecated!!!
        if not self.is_pi0_standard_gaussian():
            raise ValueError("pi0 must be a standard Gaussian distribution.")

    def assert_canonical(self):
        # Will be deprecated!!!
        if not (self.is_pi0_standard_gaussian() and (self.indepdent_coulpling==True)):
            raise ValueError('Must be the Cannonical Case: pi0 must be standard Gaussian and the data must be unpaired (independent coupling)')
        
    def set_random_seed(self, seed=None):
        # Will be deprecated!!!
        if seed is None: seed = self.seed
        self.seed = seed  
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior in certain operations (may affect performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def plot_loss_curve(self):
          plt.plot(self.loss_curve, '-.')
