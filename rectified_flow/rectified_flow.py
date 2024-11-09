#@title affine interp & rectified flow 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import sympy
import matplotlib.pyplot as plt
from collections import namedtuple

class AffineInterpSolver:
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
            raise ValueError("Exactly two of the variables (x0, x1, xt, dot_xt) must be specified.")
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
    def __init__(self, alpha=None, beta=None):
        super().__init__()
        ensure_tensor = self.ensure_tensor

        if (isinstance(alpha, str) and (alpha in ['straight', 'lerp'])) or (alpha is None and beta is None):
            # Special case for 'straight' interpolation
            alpha = lambda t: t
            beta = lambda t: 1 - t
            self.name = 'straight'

        elif isinstance(alpha, str) and alpha in ['harmonic', 'cos', 'sin', 'slerp']:
            # Special case of harmonic interpolation
            alpha = lambda t: torch.sin(t * torch.pi/2.0)
            beta = lambda t: torch.cos(t * torch.pi/2.0)
            self.name = 'harmonic'

        elif isinstance(alpha, str) and alpha in ['DDIM', 'DDPM', 'ddim', 'ddpm']:
            a = 19.9; b = 0.1
            alpha = lambda t: torch.exp(-a*(1-t)**2/4.0 - b*(1-t)/2.0)
            beta = lambda t: (1-alpha(t)**2)**(0.5)
            self.name = 'DDIM'

        else:
            self.name = 'affine'

        self.alpha = lambda t: alpha(self.ensure_tensor(t))
        self.beta = lambda t: beta(self.ensure_tensor(t))
        self.name = 'affine'

        self.solver = AffineInterpSolver()
        self.at = None; self.bt = None; self.dot_at = None; self.dot_bt = None
        self.x0 = None; self.x1 = None; self.xt = None; self.dot_xt = None

    @staticmethod
    def ensure_tensor(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x

    def match_time_dim(self, t, x):
        # enure that t is [N,...] if x is [N,...]
        t = self.ensure_tensor(t)
        while t.dim() < x.dim(): t = t.unsqueeze(-1)
        if t.shape[0] == 1: t = t.expand(x.shape[0], *t.shape[1:])
        t = t.to(x.device)
        return t

    @staticmethod
    def value_and_grad(f, input, detach=True):
        x = input.clone(); x.requires_grad_(True)
        with torch.enable_grad():
            value = f(x)
            grad, = torch.autograd.grad(value.sum(), x, create_graph=(detach==False))
        if detach: value = value.detach(); grad = grad.detach()
        return value, grad

    def get_coeffs(self, t, detach=True):
        a_t, dot_a_t = self.value_and_grad(self.alpha, t, detach=detach)
        b_t, dot_b_t = self.value_and_grad(self.beta, t, detach=detach)
        self.at = a_t; self.bt = b_t; self.dot_at = dot_a_t; self.dot_bt = dot_b_t
        return a_t, b_t, dot_a_t, dot_b_t

    def forward(self, X0, X1, t, detach=True):
        t = self.match_time_dim(t, X1)
        a_t, b_t, dot_a_t, dot_b_t = self.get_coeffs(t, detach=detach)
        Xt = a_t * X1 + b_t * X0
        dot_Xt = dot_a_t * X1 + dot_b_t * X0
        return Xt, dot_Xt

    # Solve equation: xt = at*x1+bt*x0, dot_xt = dot_at*x1+dot_bt*x0
    # Set any of the known variables, and keep the unknowns as None; the solver will fill the unknowns.
    # Example: interp.solve(t, xt=xt, dot_xt=dot_xt); print(interp.x1.shape), print(interp.x0.shape)
    def solve(self, t, x0=None, x1=None, xt=None, dot_xt=None):
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
#test_affine_interp()



class RectifiedFlow:
    def __init__(self,
                 # basic
                 D0 = None,
                 D1 = None,
                 paired = False,
                 pi0= None,
                 model=None,
                 interp=AffineInterp('straight'),
                 device = None,
                 seed = None,
                 # training
                 optimizer=None,
                 time_weight = lambda t: 0*t+1.0,
                 time_wrapper = lambda t: t,
                 criterion = lambda vt, dot_xt, xt, t, wts: torch.mean(wts * (vt - dot_xt)**2)
                 ):
        self.interp = interp
        self.pi0 = pi0
        self.D0 = D0
        self.D1 = D1
        self.paired = paired
        self.time_weight = time_weight
        self.time_wrapper = time_wrapper
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.seed = seed
        self.decoder = None # decode from hidden states 
 
        # in case we need to pass some rf configurations into the model
        if (model is not None) and (hasattr(model, 'rf_setup')) and (callable(getattr(model, 'rf_setup'))):
            model.rf_setup(self)

        self.model = model
        if (optimizer is None) and (model is not None):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.999))

        # velocity is what is actually used at inference, separate it with model to incoporate things like guidance, etc.
        self.velocity = model

        # Set the random seed if provided
        if seed is not None:
            self.set_seed(seed)

        if device is None:
            self.device = self.get_device()

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior in certain operations (may affect performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_device(self):
        model = self.model
        if isinstance(model, nn.Module):
            for param in model.parameters(): return param.device
            for buffer in model.buffers(): return buffer.device
        return torch.device('cpu')

    def ensure_tensor(self, t):
        if not isinstance(t, torch.Tensor): t = torch.tensor(t, device=self.get_device())
        return t

    def match_time_dim(self, t, x=None):
        # enure that t is [N,...] if x is [N,...]
        if x is None: x = self.D1[0]
        t = self.ensure_tensor(t)
        while t.dim() < x.dim(): t = t.unsqueeze(-1)
        if t.shape[0] == 1: t = t.expand(x.shape[0], *t.shape[1:])
        t = t.to(x.device)
        return t

    def is_pi0_zero_mean_gaussian(self):
        # Check if pi0 is a MultivariateNormal distribution
        return isinstance(self.pi0, dist.MultivariateNormal)  and torch.allclose(self.pi0.mean, torch.zeros_like(self.pi0.mean))

    def is_pi0_standard_gaussian(self):
        # Check if pi0 is Standard MultivariateNormal
        if not isinstance(self.pi0, dist.MultivariateNormal): return False
        if not torch.allclose(self.pi0.mean, torch.zeros_like(self.pi0.mean)): return False
        if not torch.allclose(self.pi0.covariance_matrix, torch.eye(self.pi0.mean.size(0), device=self.pi0.mean.device)): return False
        return True

    def assert_if_pi0_is_standard_gaussian(self):
        if not self.is_pi0_standard_gaussian():
            raise ValueError("pi0 must be a standard Gaussian distribution.")

    def assert_cannonical(self):
        if not (self.is_pi0_standard_gaussian() and (self.paired==False)):
            raise ValueError('Must be the Cannonical Case: pi0 must be standard Gaussian and the data must be unpaired (independent coupling)')

    def sample_x0(self, batch_size):
        return self.pi0.sample([batch_size]).to(self.get_device())

    def get_loss(self, X0, X1, t):
        Xt, dot_Xt = self.interp(X0, X1, t)
        vt = self.model(Xt, t)
        wts = self.time_weight(t)
        loss = self.criterion(vt, dot_Xt, Xt, t, wts)
        return loss

    def get_score_function(self, Xt, vt, t):
        self.assert_canonical()
        #t = self.match_time_dim(t, Xt)
        #at, bt, dot_at, dot_bt = self.interp.get_coeffs(t)
        #dlogp = (vt - dot_at/at * Xt) *(bt**2*dot_at/at - bt * dot_bt)**(-1)
        self.interp.solve(t, xt=Xt, dot_xt=vt)
        dlogp = - self.interp.x0/self.interp.bt
        return dlogp

    def score_function(self, Xt, t):
        t = self.match_time_dim(t, Xt)
        vt = self.model(Xt, t)
        return self.get_score_function(Xt, vt, t)

    def get_sde_params(self, vt, xt, t, e):
        t = self.match_time_dim(t, xt)
        et = e(t)
        at, bt, dot_at, dot_bt = self.interp.get_coeffs(t)
        v_t =vt * (1+et) - et * dot_at / at * xt
        sigma_t = (2 * (1-at) * dot_at/(at) * et)**(0.5)
        return v_t, sigma_t

    def profile_time_wise_loss(self, D0, D1, t_grid=torch.linspace(0,1,10), show_plot=True):
        time_wise_loss = []
        for t in t_grid:
            ts = self.match_time_dim(t, D1)
            time_wise_loss.append(self.loss(D0, D1, ts))
        loss_grid = torch.tensor(time_wise_loss)
        if show_plot:
            plt.figure(figsize=(4,4))
            plt.plot(t_grid, loss_grid, '-.')
        return t_grid, loss_grid

    @staticmethod
    def draw_sample(D, batch_size):
        if isinstance(D, dist.Distribution):
            # if D is a torch Distribution obj, call sample method
            return D.sample([batch_size])
        elif isinstance(D, torch.Tensor):
            # if D is a dataset (tensor), sample a batch.
            return D[torch.randperm(len(D))[:batch_size]]
        elif callable(D):
            return D(batch_size)
        else:
            raise NotImplementedError(f"Unsupported type: {type(D)}")

    def draw_pairs(self, batch_size=None, D0=None, D1=None, paired=None):
        if D0 is None: D0 = self.D0
        if D1 is None: D1 = self.D1
        if paired is None: paired = self.paired
        if batch_size is None: batch_size = self.batch_size
        draw_sample = self.draw_sample

        if not paired:
            # Draw unpaired data
            X1 = draw_sample(D1, batch_size)
            X0 = draw_sample(D0, batch_size)
        else:
            # Draw minibatch of paired data, used for reflow
            idx = torch.randperm(len(D1))[:batch_size]
            X0, X1 = D0[idx], D1[idx]
        return X0, X1

    def draw_time(self, batch_size):
        t = self.time_wrapper(torch.rand((batch_size, 1)).to(self.get_device()))
        return t

    def train(self, num_iterations = 100, batch_size=64, D0=None, D1=None, optimizer=None):
        if D0 is None: D0 = self.D0
        if D1 is None: D1 = self.D1
        if optimizer is None: optimizer = self.optimizer

        draw_pairs = self.draw_pairs
        interp = self.interp
        model = self.model

        self.loss_curve = []
        for i in range(num_iterations):
            optimizer.zero_grad()

            # draw data pairs
            X0, X1 = draw_pairs(D0=D0, D1=D1, batch_size=batch_size)

            # draw time
            t = self.draw_time(batch_size)

            # loss
            loss = self.get_loss(X0, X1, t)
            loss.backward()
            optimizer.step()

            # track loss
            self.loss_curve.append(loss.item())

    def plot_loss_curve(self):
          plt.plot(self.loss_curve, '-.')


