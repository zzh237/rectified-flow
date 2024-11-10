#@title rectified_flow_samplers
import torch
import matplotlib.pyplot as plt
from collections import namedtuple

class Sampler:
    def __init__(self, rf, x0=None, num_steps=100, record_traj_period=1, num_points=100, seed = None):
        self.rf = rf
        self.velocity = rf.velocity
        self.num_steps = num_steps
        self.record_traj_period = record_traj_period
        self.results = None
        self.alpha = self.rf.interp.alpha
        self.beta = self.rf.interp.beta

        # seed
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

        # initialization
        self.x0 = x0 if x0 is not None else rf.sample_x0(num_points)
        self.num_points = num_points if x0 is None else x0.shape[0]

    def step(self, xt, t, step_size):
        raise NotImplementedError("Sampler subclass must implement the step method.")

    def sample(self):

        xt = self.x0.clone()
        info_t = None

        trajectories = [xt]
        info = [info_t]
        time = [0]

        with torch.no_grad():
            for step in range(self.num_steps):
                t = step / self.num_steps
                step_size = 1 / self.num_steps
                t_next = min(t + step_size, 1)

                # Call the specific step function defined in the subclass
                xt, info_t = self.step(xt, t, t_next, info=info_t)

                time.append(t_next)
                trajectories.append(xt)
                info.append(info_t)

            trajectories = torch.stack(trajectories)

        Results = namedtuple('Results', ['xt', 'trajectories', 'info', 'time'])
        self.results = Results(xt, trajectories, info, time)

    def plot_2d_results(self, num_trajectories = 50, markersize=3, dimensions=[0,1], alpha_trajectories=0.5, alpha_generated_points=1, alpha_true_points=1):
        dim0 = dimensions[0]; dim1 = dimensions[1]
        xtraj = self.results.trajectories.detach().cpu().numpy()
        try:
            if self.rf.D1 is not None:
                plt.plot(self.rf.D1[:, dim0].detach().cpu().numpy() , self.rf.D1[:, dim1].detach().cpu().numpy() , '.', label='D1', markersize=markersize, alpha=alpha_true_points)
            if self.rf.D0 is not None:
                plt.plot(self.rf.D0[:, dim0].detach().cpu().numpy() , self.rf.D0[:, dim1].detach().cpu().numpy() , '.', label='D0', markersize=markersize,alpha=alpha_true_points)
        except:
            pass

        plt.plot(xtraj[-1][:, dim0], xtraj[-1][:, dim1], 'r.', label='gen', markersize=markersize, alpha = alpha_generated_points)
        plt.plot(xtraj[:, :num_trajectories, dim0], xtraj[:, :num_trajectories, dim1], '--g', alpha=alpha_trajectories)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


class EulerSampler(Sampler):
    def step(self, xt, t, t_next, info=None):
        t_ones = t * torch.ones(xt.shape[0], 1).to(self.x0.device)
        v_t = self.velocity(xt, t_ones.squeeze())
        x1_pred = xt + (1 - t) * v_t
        xt = xt + (t_next - t) * v_t
        return xt, x1_pred


class CurvedSampler(Sampler):
    def step(self, xt, t, t_next, info=None):
        t = self.rf.match_time_dim(t, xt) 
        vt = self.velocity(xt, t.squeeze())
        # given xt, and dot_xt = vt, find the corresponding end points x0, x1
        self.rf.interp.solve(t, xt=xt, dot_xt=vt)
        x1_pred = self.rf.interp.x1
        x0_pred = self.rf.interp.x0
        # interplate to find x_{t_next} 
        self.rf.interp.solve(t_next, x0=x0_pred, x1=x1_pred)
        xtnext = self.rf.interp.xt
        return xtnext, x1_pred  

class OverShootingSampler(Sampler):
    def __init__(self, rf, x0=None, num_steps=100, record_traj_period=1, num_points=100,  seed = None,
                 c=1.0,
                 overshooting_method='t+dt*(1-t)'
                 ):
        super().__init__(rf, x0, num_steps, record_traj_period, num_points, seed)
        self.c = c

        # Define overshooting method
        if callable(overshooting_method):
            self.overshooting = overshooting_method
        elif isinstance(overshooting_method, str):
            self.overshooting = eval(f"lambda t, dt: {overshooting_method}")
        else:
            raise NotImplementedError("Invalid overshooting method provided")

        assert (rf.is_pi0_zero_mean_gaussian() and rf.paired==False), "pi0 must be a zero-mean Gaussian distribution, and the coupling must be independent."

    def step(self, xt, t, t_next, info=None):
        step_size = 1 / self.num_steps
        step_size_overshoot = step_size * self.c

        # Calculate overshoot time and enforce constraints
        t_overshoot = min(self.overshooting(t_next, step_size_overshoot), 1)

        if t_overshoot < t_next: raise ValueError("t_overshoot cannot be smaller than t_next.") 

        # Predict x1 and store it
        vt = self.velocity(xt,  self.rf.match_time_dim(t, xt).squeeze())

        # Advance to t_overshoot with ODE 
        xt_overshoot = xt + (t_overshoot - t) * vt

        # Apply noise to step back to t_next 
        at = self.alpha(t_next) / self.alpha(t_overshoot)
        bt = (self.beta(t_next)**2 - (at * self.beta(t_overshoot))**2)**0.5
        noise = self.rf.sample_x0(xt.shape[0])
        xt = xt_overshoot * at +  noise * bt
        return xt, vt


class SDESampler(Sampler):
    def __init__(self, rf, x0=None, num_steps=100, record_traj_period=1, num_points=100,  seed = None,
                 e = lambda t: 1.0):
        super().__init__(rf, x0, num_steps, record_traj_period, num_points, seed)
        self.e = e

        assert (rf.is_pi0_standard_gaussian() and rf.paired==False), "pi0 must be a standard Gaussian distribution, and the coupling must be independent."

    def step(self, xt, t, t_next, info=None):
        t_ones = t * torch.ones(xt.shape[0], 1).to(xt.device)
        t_eps = 1e-12
        t_ones = torch.clamp(t_ones, t_eps, 1 - t_eps)
        step_size = t_next - t

        # Calculate alpha and beta values and their gradients
        e_t = self.e(t_ones)
        a_t, b_t, dot_a_t, dot_b_t = self.rf.interp.get_coeffs(t_ones)

        # Model prediction and adjusted velocity
        v_t = self.velocity(xt, t_ones.squeeze())
        v_adj_t = (1 + e_t) * v_t - e_t * dot_a_t / a_t * xt
        sigma_t = torch.sqrt(2 * (b_t**2 * dot_a_t / a_t - dot_b_t * b_t) * e_t)

        # Predicted x1 and update to xt with added noise
        x1_pred = xt + (1 - t) * v_t
        noise = self.rf.sample_x0(xt.shape[0])
        xt = xt + step_size * v_adj_t + sigma_t * torch.sqrt(torch.tensor(step_size)) * noise

        return xt, x1_pred
