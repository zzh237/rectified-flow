#@title affine interp & rectified flow, CouplingDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import sympy
import matplotlib.pyplot as plt
from collections import namedtuple
import warnings
from torch.utils.data import Dataset, DataLoader

class CouplingDataset(Dataset):
    def __init__(self, noise=None, data=None, labels=None, independent_coupling=True):
        """
        Initialize the dataset with noise (D0), data (D1), and optional labels.

        Args:
            noise: Tensor, distribution, or function for generating noise samples. If None, defaults to a standard
                   multivariate normal distribution with the same dimensions as each sample of data.
            data: Tensor or distribution for data samples.
            labels: Optional tensor for labels associated with data samples.
            data_size: Default length for the dataset if data is not a tensor.
            independent_coupling: If True, samples D0 and D1 independently; otherwise, pairs them.
        """
        self.D1 = data
        self.D0 = noise if noise is not None else self._set_default_noise()
        self.labels = labels
        self.independent_coupling = independent_coupling
        self.paired = not independent_coupling
        self._varlidate_inputs()
        self.default_dataset_length = 10000

    def randomize_D0_index_if_needed(self, index):
        """Randomize indices for D0 if pairing=False and D0 is Tensor."""
        if not self.paired and isinstance(self.D0, torch.Tensor):
            return torch.randint(0, len(self.D0), (1,)).item()
        else:
            return index

    def _set_default_noise(self):
        """Set up default noise as a standard normal matching the sample shape of D1."""
        data_shape = self.draw_sample(self.D1, 0).shape
        return dist.Normal(torch.zeros(data_shape), torch.ones(data_shape))

    @staticmethod
    def draw_sample(D, index):
        """
        Draw a sample based on the type of D (tensor, distribution, or callable).
        Returns D[index] if D is a tensor, otherwise a sample from D (index ignored).
        """
        if isinstance(D, dist.Distribution):
            return D.sample([1]).squeeze(0)
        elif isinstance(D, torch.Tensor):
            return D[index]
        elif callable(D):
            return D(1)
        else:
            raise NotImplementedError(f"Unsupported type: {type(D)}")

    def __len__(self):
        """Return the length of D1 if it's a tensor, otherwise default."""
        return len(self.D1) if isinstance(self.D1, torch.Tensor) else self.default_dataset_length

    def __getitem__(self, index):
        """Retrieve a sample from D0 and D1, and labels if provided."""
        X0 = self.draw_sample(self.D0, self.randomize_D0_index_if_needed(index))
        X1 = self.draw_sample(self.D1, index)
        if self.labels is not None:
            label = self.draw_sample(self.labels, index)
            return X0, X1, label
        else:
            return X0, X1

    # Input validation based on pairing
    def _varlidate_inputs(self):
        if self.paired:
            if self.labels is None:
                assert isinstance(self.D0, torch.Tensor) and isinstance(self.D1, torch.Tensor) and len(self.D0) == len(self.D1), \
                    "D0 and D1 must be tensors of the same length when paired is True."
            else:
                assert isinstance(self.D0, torch.Tensor) and isinstance(self.D1, torch.Tensor) and isinstance(self.labels, torch.Tensor) \
                      and len(self.D0) == len(self.D1) == len(self.labels), \
                    "D0, D1, and labels must be tensors of the same length when paired is True."
        else:
            if self.labels is not None:
                assert isinstance(self.D1, torch.Tensor) and isinstance(self.labels, torch.Tensor) and len(self.D1) == len(self.labels), \
                    "D1 and labels must be tensors of the same length when labels are given."

# Testing code
def test_coupling_dataset():
    # Test data tensor with more than two dimensions
    data = torch.randn(5, 3,4)  # 100 samples, each of shape (3, 4, 4)
    labels = torch.randint(0, 2, (5,))  # Binary labels for testing

    # Case 1: Independent dataset with default noise as standard normal
    independent_dataset_default_noise = CouplingDataset(data=data, independent_coupling=True)
    independent_dataloader_default_noise = DataLoader(independent_dataset_default_noise, batch_size=2)

    print("Testing independent dataset with default standard normal noise:")
    for X0, X1 in independent_dataloader_default_noise:
        #assert X0.shape == X1.shape == (10, 3, 4, 4), "Independent samples should have the same shape"
        print("Independent batch with default standard normal noise:", X0.shape, X1.shape)

    # Case 2: Independent dataset with custom noise distribution
    noise_dist = dist.Normal(torch.zeros(3), torch.ones(3))  # Distribution matching (3, 4, 4) shape
    data = torch.randn(5, 3)  # 100 samples, each of shape (3, 4, 4)
    independent_dataset_dist = CouplingDataset(noise=noise_dist, data=data, independent_coupling=True)
    independent_dataloader_dist = DataLoader(independent_dataset_dist, batch_size=2, drop_last=True)

    print("\nTesting independent dataset with custom noise distribution:")
    for X0, X1 in independent_dataloader_dist:
        #assert X0.shape == X1.shape == (10, 3, 4, 4), "Independent samples should have the same shape"
        print("Independent batch with noise from distribution:", X0.shape, X1.shape)

# Run tests
if __name__ == '__main__':
    test_coupling_dataset()



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

    def match_time_dim(self, t, x):
        # enure that t is [N,1,1,..] if x is [N,...]
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

if __name__ == "__main__":
    test_affine_interp()

class RectifiedFlow:
    def __init__(self,
                 # basic
                 dataset= None,
                 pi0= None,
                 interp=AffineInterp('straight'),
                 model=None,
                 device = None,
                 seed = None,
                 # training
                 optimizer=None,
                 time_weight = lambda t: 0*t+1.0,
                 time_wrapper = lambda t: t,
                 criterion = lambda vt, dot_xt, xt, t, wts: torch.mean(wts * (vt - dot_xt)**2),
                 # training loops
                 num_iterations=100,
                 num_epochs=None,
                 batch_size=64,
                 ):
        # interp
        self.interp = interp

        # data
        self.dataset = dataset
        self.D0 = dataset.D0
        self.D1 = dataset.D1
        self.labels = dataset.labels
        self.independent_coupling = dataset.independent_coupling
        self.paired = not self.independent_coupling
        self.pi0 = pi0 if pi0 is not None else dataset.D0

        # training & model
        self.time_weight = time_weight
        self.time_wrapper = time_wrapper
        self.criterion = criterion
        self.device = device
        self.seed = seed
        self.decoder = None # decode from hidden states
        self.model = model
        self.optimizer = optimizer
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # defaults
        self.config_model()
        self.set_default_optimizer()
        self.set_random_seed()
        self.set_device()

    # decouple velocity and model if needed 
    def velocity(self, *args, **kwargs):
        # Remove 'label' from kwargs if rf.dataset.labels = None
        if self.labels is None and 'labels' in kwargs and kwargs['labels'] is None: 
            del kwargs['labels']        
        return self.model(*args, **kwargs)


    # in case we need to pass some rf configurations into the model
    def config_model(self):
        if (self.model is not None) and (hasattr(self.model, 'rectifiedflow_setup')) and (callable(getattr(self.model, 'rectifiedflow_setup'))):
            self.model.rectifiedflow_setup(self)

    def set_default_optimizer(self):
        if (self.optimizer is None) and (self.model is not None):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.0, betas=(0.9, 0.999))

    def set_random_seed(self, seed=None):
        if seed is None: seed = self.seed
        self.seed = seed  
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior in certain operations (may affect performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def set_device(self):
        if self.device is None:
            device = torch.device('cpu')
            if isinstance(self.D1, torch.Tensor):
                device = self.D1.device
            if isinstance(self.model, nn.Module):
                for param in self.model.parameters():
                    device = param.device
                    break
            self.device = device
            return device

    def ensure_tensor(self, t):
        if not isinstance(t, torch.Tensor): t = torch.tensor(t, device=self.device)
        return t

    def match_time_dim(self, t, x=None):
        # enure that t is [N,...] if x is [N,...]
        if x is None: x = self.D1[0]
        t = self.ensure_tensor(t)
        while t.dim() < x.dim(): t = t.unsqueeze(-1)
        if t.shape[0] == 1: t = t.expand(x.shape[0], *t.shape[1:])
        t = t.to(x.device)
        return t

    def sample_x0(self, batch_size):
        return self.pi0.sample([batch_size]).to(self.device)

    def get_loss(self, X0, X1, *args, **kwargs):
        t = self.draw_time(X0.shape[0])
        Xt, dot_Xt = self.interp(X0, X1, t)
        vt = self.model(Xt, t, *args, **kwargs)
        wts = self.time_weight(t)
        loss = self.criterion(vt, dot_Xt, Xt, t, wts)
        return loss

    # for D0~Normal(0,I), Dlogpt(Xt) = -E[X0|Xt]/bt
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

    def draw_time(self, batch_size):
        t = self.time_wrapper(torch.rand((batch_size, 1)).to(self.device))
        return t

    def train(self, num_iterations=100, num_epochs=None, batch_size=64, D0=None, D1=None, optimizer=None, shuffle=True):
        if optimizer is None: optimizer = self.optimizer
        if num_iterations is None: num_iterations = self.num_iterations
        if num_epochs is not None: num_epochs = self.num_epochs
        self.loss_curve = []

        # Set up dataloader
        dataloader = torch.utils.data.DataLoader(self.data, batch_size=batch_size, shuffle=shuffle)
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
        # Check if pi0 is a zero-mean Gaussian distribution
        case1 = isinstance(self.pi0, dist.MultivariateNormal)  and torch.allclose(self.pi0.mean, torch.zeros_like(self.pi0.mean))
        case2 = isinstance(self.pi0, dist.Normal)  and torch.allclose(self.pi0.mean, torch.zeros_like(self.pi0.loc))
        return case1 or case2

    def is_pi0_standard_gaussian(self):
        # Check if pi0 is Standard Gaussian
        case1 = isinstance(self.pi0, dist.MultivariateNormal)  \
                and torch.allclose(self.pi0.mean, self.pi0.mean*0) \
                and torch.allclose(self.pi0.covariance_matrix, torch.eye(self.pi0.mean.size(0), device=self.pi0.mean.device))
        case2 = isinstance(self.pi0, dist.Normal)  \
                and torch.allclose(self.pi0.mean, self.pi0.mean*0) \
                and torch.allclose(self.pi0.variance, self.pi0.variance*0 + 1)
        return case1 or case2

    def assert_if_pi0_is_standard_gaussian(self):
        if not self.is_pi0_standard_gaussian():
            raise ValueError("pi0 must be a standard Gaussian distribution.")

    def assert_canonical(self):
        if not (self.is_pi0_standard_gaussian() and (self.indepdent_coulpling==True)):
            raise ValueError('Must be the Cannonical Case: pi0 must be standard Gaussian and the data must be unpaired (independent coupling)')

    def plot_loss_curve(self):
          plt.plot(self.loss_curve, '-.')
