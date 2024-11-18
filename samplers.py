#@title rectified_flow_samplers
import torch
import matplotlib.pyplot as plt
from collections import namedtuple

class Sampler:
    def __init__(self, rf, x0=None, labels = None, num_steps=100, record_traj_period=1, num_points=100, seed = None, time_grid = None):

        # read inputs 
        self.rf = rf
        self.xt = x0
        self.num_steps = num_steps 
        self.labels = labels 
        self.record_traj_period = record_traj_period 
        self.num_points = num_points 
        self.seed = seed 
        self.time_grid = time_grid 
        self.maximum_num_steps = 100000 

        # prepare 
        self.set_random_seed() 
        self.process_args()
        self.set_default_time_grid() 
        self.initialize() 
        self.set_recording_time_grid() 
        self.results = None

    def set_random_seed(self, seed=None):
        if seed is None: seed = self.seed
        self.seed = seed  
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior in certain operations (may affect performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def process_args(self):
        # overwrite num_points if needed
        if self.xt is not None:
            self.num_points = self.xt.shape[0]
        elif self.labels is not None:
            self.num_points = self.labels.shape[0]

        # overwrite num_steps if needed
        if isinstance(self.time_grid, torch.Tensor):
            self.num_steps = len(self.time_grid)-1

    # initialize x0 by rf.sample_x0() 
    def initialize_xt(self):
        if self.xt is None:
            self.xt = self.rf.sample_x0(self.num_points)

    # use uniform time grid of size num_steps by default 
    def set_default_time_grid(self):
        if self.time_grid is None:
              self.time_grid = torch.linspace(0, 1, self.num_steps + 1)

    # the time grid on which we record the results 
    def set_recording_time_grid(self):
        self.recording_time_grid = list(range(0, self.num_steps + 1, self.record_traj_period)) + ([self.num_steps] if self.num_steps % self.record_traj_period != 0 else [])

    # intialize xt, t, t_next
    def initialize(self):

        self.step_count = 0
        self.t = self.time_grid[0]
        self.t_next = self.time_grid[1]

        # initalize x0 
        self.initialize_xt()

        # recording trajectories 
        self.trajectories = [self.xt]
        self.time_points = [self.t] 

    # use self.labels by default
    def get_velocity(self): 
        xt, t, labels = self.xt, self.t, self.labels          
        t = self.rf.match_time_dim(t, xt) 
        return self.rf.velocity(xt, t, labels=labels) 

    def step(self):
        raise NotImplementedError("Sampler subclass must implement the step method.")

    def set_next_time_point(self):
        self.t_next = self.time_grid[self.step_count+1]
        self.step_count = self.step_count+1

    def stop(self):
        return (self.t >= 1.0-1e-6) or (self.step_count >= self.maximum_num_steps)

    def record_trajectories(self):
        if self.step_count in self.recording_time_grid:
            self.trajectories.append(self.xt)
            self.time_points.append(self.t)

    def record_other_information(self):
        pass

    def sample(self):

        # initialize
        self.initialize()

        with torch.no_grad():
            while True:

                # update t and xt
                self.step()

                # record information
                self.record_trajectories()
                self.record_other_information()

                # stop criterion
                if self.stop(): break

                # update t_next
                self.set_next_time_point()


        return self;

    def plot_2d_results(self, num_trajectories = 50, markersize=3, dimensions=[0,1], alpha_trajectories=0.5, alpha_generated_points=1, alpha_true_points=1):
        dim0 = dimensions[0]; dim1 = dimensions[1]
        xtraj = torch.stack(self.trajectories).detach().cpu().numpy()
        try:
            if isinstance(self.rf.D1, torch.Tensor):
                plt.plot(self.rf.D1[:, dim0].detach().cpu().numpy() , self.rf.D1[:, dim1].detach().cpu().numpy() , '.', label='D1', markersize=markersize, alpha=alpha_true_points)
            plt.plot(xtraj[0][:, dim0], xtraj[0][:, dim1], '.', label='D0', markersize=markersize,alpha=alpha_true_points)
        except:
            pass

        plt.plot(xtraj[-1][:, dim0], xtraj[-1][:, dim1], 'r.', label='gen', markersize=markersize, alpha = alpha_generated_points)
        plt.plot(xtraj[:, :num_trajectories, dim0], xtraj[:, :num_trajectories, dim1], '--g', alpha=alpha_trajectories)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


class EulerSampler(Sampler):
    def step(self):
        t, t_next, xt = self.t, self.t_next, self.xt
        vt = self.get_velocity()
        self.xt = xt + (t_next - t) * vt
        self.t = t_next

class CurvedEulerSampler(Sampler):
    def step(self):

        t, t_next, xt = self.t, self.t_next, self.xt
        vt = self.get_velocity()

        self.rf.interp.solve(t, xt=xt, dot_xt=vt)
        x1_pred = self.rf.interp.x1
        x0_pred = self.rf.interp.x0
        # interplate to find x_{t_next}
        self.rf.interp.solve(t_next, x0=x0_pred, x1=x1_pred)
        xt = self.rf.interp.xt

        self.xt = xt 
        self.t = t_next

class NoiseRefreshSampler(Sampler):
    def __init__(self, *args, noise_replacement_rate = lambda t: 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_replacement_rate = noise_replacement_rate # should be in [0,1]
        assert (self.rf.paired==False and self.rf.is_pi0_zero_mean_gaussian()==True), 'pi0 must be a zero mean gaussian and must use indepdent coupling'

    def step(self):

        t, t_next, xt = self.t, self.t_next, self.xt
        vt = self.get_velocity()

        # given xt, and dot_xt = vt, find the corresponding end points x0, x1
        self.rf.interp.solve(t, xt=xt, dot_xt=vt)
        x1_pred = self.rf.interp.x1
        x0_pred = self.rf.interp.x0

        # randomize x0_pred by replacing part of it with new noise
        noise = rf.sample_x0(xt.shape[0])
        x0_pred_noise = (1-self.noise_replacement_rate(t)**2)**(0.5) * x0_pred + noise * self.noise_replacement_rate(t)

        # interplate to find xt at t_next
        self.rf.interp.solve(t_next, x0=x0_pred_noise, x1=x1_pred)
        xtnext = self.rf.interp.xt

        self.xt = xtnext
        self.t = t_next



class OverShootingSampler(Sampler):
    def __init__(self, rf,
                 c=1.0,
                 overshooting_method='t+dt*(1-t)', **kwargs):
        super().__init__(rf, **kwargs)
        self.c = c

        # Define overshooting method
        if callable(overshooting_method):
            self.overshooting = overshooting_method
        elif isinstance(overshooting_method, str):
            self.overshooting = eval(f"lambda t, dt: {overshooting_method}")
        else:
            raise NotImplementedError("Invalid overshooting method provided")

        assert (rf.is_pi0_zero_mean_gaussian() and rf.paired==False), "pi0 must be a zero-mean Gaussian distribution, and the coupling must be independent."

    def step(self):
        t, t_next, xt = self.t, self.t_next, self.xt
        vt = self.get_velocity()
        alpha = self.rf.interp.alpha
        beta = self.rf.interp.beta

        # Calculate overshoot time and enforce constraints
        t_overshoot = min(self.overshooting(t_next,  (t_next - t)*self.c), 1)
        if t_overshoot < t_next: raise ValueError("t_overshoot cannot be smaller than t_next.")

        # Advance to t_overshoot with ODE
        xt_overshoot = xt + (t_overshoot - t) * vt

        # Apply noise to step back to t_next
        at = alpha(t_next) / alpha(t_overshoot)
        bt = (beta(t_next)**2 - (at * beta(t_overshoot))**2)**0.5
        noise = self.rf.sample_x0(xt.shape[0])
        xt = xt_overshoot * at +  noise * bt

        self.xt = xt
        self.t = t_next 


class SDESampler(Sampler):
    def __init__(self, rf, e = lambda t: 1.0, **kwargs):
        super().__init__(rf, **kwargs)
        self.e = e

        assert (rf.is_pi0_standard_gaussian() and rf.paired==False), "pi0 must be a standard Gaussian distribution, and the coupling must be independent."

    def step(self):
        t, t_next, xt = self.t, self.t_next, self.xt
        vt = self.get_velocity()

        t_ones = t * torch.ones(xt.shape[0], 1).to(xt.device)
        t_eps = 1e-12
        t_ones = torch.clamp(t_ones, t_eps, 1 - t_eps)
        step_size = t_next - t

        # Calculate alpha and beta values and their gradients
        e_t = self.e(t_ones)
        a_t, b_t, dot_a_t, dot_b_t = self.rf.interp.get_coeffs(t_ones)

        # Model prediction and adjusted velocity
        v_adj_t = (1 + e_t) * vt - e_t * dot_a_t / a_t * xt
        sigma_t = torch.sqrt(2 * (b_t**2 * dot_a_t / a_t - dot_b_t * b_t) * e_t)

        # Predicted x1 and update to xt with added noise
        x1_pred = xt + (1 - t) * vt
        noise = self.rf.sample_x0(xt.shape[0])
        xt = xt + step_size * v_adj_t + sigma_t * (step_size)**(0.5) * noise

        self.xt = xt
        self.t = t_next 
