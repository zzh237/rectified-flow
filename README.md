<div align="center">

<img src="assets/logo.svg" alt="Logo" style="width: 80%; height: auto;">

[![Blog](https://img.shields.io/badge/blog-blue)](https://rectifiedflow.github.io)
[![Book](https://img.shields.io/badge/book-blue)](https://arxiv.org/pdf/2209.03003.pdf)
[![Email](https://img.shields.io/badge/rectified_flow-email-blue)](mailto:rectifiedflow@googlegroups.com)

______________________________________________________________________

</div>


## Overview

**RectifiedFlow** is a repository that offers a *unified* and *minimal* codebase for implementing a variety of diffusion and flow models, including [Rectified Flow](https://arxiv.org/pdf/2209.03003.pdf), [DDPM](https://arxiv.org/abs/2006.11239.pdf), [DDIM](https://arxiv.org/abs/2010.02502.pdf), and more. By adopting a unified Ordinary Differential Equation (ODE) perspective, **RectifiedFlow** provides a streamlined and convenient framework for training and inference, tailored for research purposes. This PyTorch-based library includes:

- **Unified Training and Inference**: Seamlessly train and infer rectified flow (flow matching) and diffusion models from a single, coherent ODE perspective.
  
- **Interactive Tutorials**: Engage with beginner-friendly tutorials that offer hands-on experience with rectified flows. Learn how to effortlessly transform rectified flow models into other models like DDIM and demonstrate their equivalence.
  
- **Comprehensive Tools**: Access a robust set of tools for studying Rectified Flow models, including interpolation methods and ODE/SDE solvers. Designed with the ODE framework in mind, these tools are both easy to comprehend and use.
  
- **Support for State-of-the-Art Models**: Utilize support for cutting-edge open-source models such as Flux.1-dev, ensuring that you can implement advanced tasks with surprising ease. Demos demonstrate how to perform sophisticated tasks like image editing with minimal effort.

Whether you are a researcher exploring the frontiers of generative modeling or a practitioner seeking to deepen your understanding through comprehensive tutorials, **RectifiedFlow** provides the essential resources and functionalities to advance your projects with confidence and ease.

---

# Contents
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [A Quick Walkthrough](#a-quick-walkthrough)
  - [Interactive Tutorials](#interactive-tutorials)
  - [Training Pipelines](#training-pipelines)
- [Customize Your Component](#customize-your-component)
  - [Wrapping a New Velocity](#wrapping-a-new-velocity)
  - [Interpolation](#interpolation)
  - [Training Utilities](#training-utilities)
  - [Customized Samplers](#customized-samplers)
- [License](#license)


# Installation

Please run the following commands in the given order to install the dependency.


```
conda create -n rf python=3.10
conda activate rf
git clone https://github.com/lqiang67/rectified-flow.git
cd rectified-flow
pip install -r requirements.txt
```

Then install the `rectified-flow` package:

```
pip install -e .
```

# Getting Started

## A Quick Walkthrough
The `RectifiedFlow` class serves as an **intermediary** for your training and inference processes. Each different velocity field should instantiate a separate `RectifiedFlow` class.

```python
from rectified_flow.rectified_flow import RectifiedFlow

model = YourVelocityFieldModel()

# Initialize RectifiedFlow with custom settings
rectified_flow = RectifiedFlow(
    data_shape=(32, 32),
    velocity_field=model,
    interp="straight",
    source_distribution="normal",
    is_independent_coupling=True,
    train_time_distribution="uniform",
    train_time_weight="uniform",
    criterion="mse",
    device=device,
)

# Or use the default settings
rectified_flow = RectifiedFlow(
    data_shape=(32, 32),
    velocity_field=model,
    device=device,
)
```

During training, you can easily compute the predefined loss by passing your target data samples `x_1`. If samples from source distribution `x_0` is not provided, it will be sampled by default. The `RectifiedFlow` class supports various pre-specified loss functions and interpolation methods, and it calculates the loss accordingly.

```python
loss = rectified_flow.get_loss(x_0=None, x_1=x_1, **kwargs)
```
This is implemented by:
```python
# Step 1: Interpolation
x_t, dot_x_t = self.get_interpolation(x_0, x_1, t)

# Step 2: Velocity Calculation
v_t = self.get_velocity(x_t, t, **kwargs)

# Step 3: Time Weights
time_weights = self.train_time_weight(t)

# Step 4: Loss Computation
return self.criterion(
    v_t=v_t,
    dot_x_t=dot_x_t,
    x_t=x_t,
    t=t,
    time_weights=time_weights,
)

```

For sampling, import the desired sampler class and pass the `RectifiedFlow` instance to it.
```python
from rectified_flow.samplers import SDESampler

sde_sampler = SDESampler(rectified_flow=rectified_flow)

sde_sampler.sample_loop(
    num_samples=128,
    num_steps=100,
    seed=0,
)

traj = sde_sampler.trajectories
img = traj[-1]
```

## Interactive Tutorials

1. **Introduction with 2D Toy**: This [notebook](examples/train_2d_toys.ipynb) provides an example using a 2D toy to illustrate the basic concepts of Rectified Flow. It covers the interpolation process $\{X_t\}$, the rectified flow $\{Z_t\}$ with velocity $\mathbb{E}[\dot{X}_t \mid X_t]$, and Reflow $\{Z^1_t\}$.
2. **Samplers**: This [notebook](examples/samplers_2d_toys.ipynb) explores the samplers provided in this repository using a 2D toy example. It illustrates the concepts and usage of various samplers such as `CurvedEuler`, `Overshooting`, and `SDESampler`. Additionally, it demonstrates how to customize your own sampler by inheriting from the `Sampler` base class. And discusses the implications of using stochastic samplers.
3. **Interpolation**: This [notebook](examples/interpolation_conversion.ipynb) first illustrates the idea that different interpolations $\{X_t\}$ can be converted from one another and presents a simple implementation to achieve this conversion. It also reveals the interesting fact that the very same transformation applies to $\{Z_t\}$, along with a few notable findings.
4. **Flux**: We provide a [notebook](examples/inference_flux_dev.ipynb) that shows how to easily interact with the wrapped Flux model using different samplers. Additionally, another [notebook](examples/editing_flux_dev.ipynb) demonstrates how to perform image editing task with Flux. All in a straightforward and friendly manner.

## Training Pipelines

We provide **[Diffusers](https://github.com/huggingface/diffusers)-style** training scripts for [UNet](https://github.com/NVlabs/edm). and [DiT](https://github.com/facebookresearch/DiT) in this [directory](rectified_flow/pipelines). The training scripts utilizes **[Accelerate](https://github.com/huggingface/accelerate)** for multi-GPU training.

**Results Using this Training Scripts**:

- **UNet CIFAR10**: Trained for $500 \text{k}$ iterations with `batch_size=128`. You can download the model [here](). $\text{FID}_{50\text{K}}=4.308$.
- **DiT CIFAR10**: Trained for $1000 \text{k}$ iterations with `batch_size=128`. You can download the model [here](). $\text{FID}_{50\text{K}}=3.678$.

**Loading a Pretrained Model**:

To construct a model from a pretrained checkpoint, simply run the following code:

```python
from rectified_flow.models.dit import DiT

model = DiT.from_pretrained(save_directory="PATH_TO_MODEL", filename="dit", use_ema=True).to(device)
```

# Customize Your Component

## Wrapping a New Velocity

The `velocity_field` argument in the `RectifiedFlow` class accepts a neural network or any callable function that takes $x_t$ and $t$ as inputs. If you need to reparameterize the model or reverse the direction of the generating ODE time (from $t = 1 \to 0$ to $t = 0 \to 1$), we recommend using a velocity field wrapper to simplify the process.

1. **Reversing the Time Direction**  
   In scenarios like Flux, where the velocity transitions from $\pi_1$ to $\pi_0$ for image generation, reversing the ODE time direction is often required. A simple wrapper achieves this.

   ```python
   # Reverse ODE time direction
   velocity = lambda x_t, t: -model(x_t, 1.0 - t)
   ```

2. **Reparameterizing for Noise Prediction**  
   Some works parameterize the model to predict noise instead of velocity. Using an `AffineInterpSolver`, you can automatically convert noise predictions into velocity predictions, bypassing the complexity of handling DDIM coefficients.

   ```python
   # Convert noise prediction to velocity prediction
   velocity = lambda x_t, t: self.interp.solve(t=t, x_t=x_t, x_0=model(x_t, t)).dot_x_t
   ```
In Rectified Flow, we assume that when $t = 0$, $X_0 \sim \pi_0$ represents the **source distribution**, and when $t = 1$, $X_1 \sim \pi_1$ represents the **target distribution**, such as an image distribution.

## Interpolation

The `AffineInterp` class manages the affine interpolation between the source distribution $\pi_0$ and the target distribution $\pi_1$. It offers two primary features:

1. **Automatic Interpolation Handling**: By providing $\alpha_t$ and $\beta_t$ functions (along with optional $\dot{\alpha}_t$ and $\dot\beta_t$), `AffineInterp` computes the interpolated state $X_t$ and its time derivative $\dot X_t$. If the derivatives functions $\dot \alpha_t, \dot\beta_t$ are not supplied, they are calculated automatically using Pytorch automatic differentiation.
   ```python
   from rectified_flow.flow_components import AffineInterp

   alpha_function = lambda t: torch.sin(a * t) / torch.sin(a)
   beta_function = lambda t: torch.sin(a * (1 - t)) / torch.sin(a)

   interp = AffineInterp(alpha=alpha_function, beta=beta_function)

   x_t, dot_x_t = interp.forward(x_0, x_1, t)
   ```
2. **Automatic Solving of Unknown Variables**: Given any two of the four variables ($X_0,X_1,X_t,\dot X_t$), the class can automatically solve for the remaining unknowns using precomputed symbolic solvers. This feature is very convenient when computing certain common quantities, such as estimating $\hat X_0$ and $\hat X_1$ given $X_t$ and $v(X_t, t)$.

    ```python
    # Solve for x_0 and x_1 given x_t and dot_x_t
    interp.solve(t=t, x_t=x_t, dot_x_t=velocity)
    print(interp.x_0, interp.x_1)
    ```

## Training Utilities

To tailor the training process to your specific requirements, you can customize these utilities by inheriting from their base classes and overriding their methods. Once customized, simply pass the instances to the `RectifiedFlow` class during initialization.

**Example: Custom Weighting Scheme**
```python
from rectified_flow.flow_components import TrainTimeWeight

class CustomTimeWeight(TrainTimeWeight):
    def __init__(self):
        super().__init__()

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        wts = torch.exp(t)
        return wts

# Initialize with custom exponential weighting
custom_time_weight = CustomTimeWeight()
```

## Customized Samplers

To create custom samplers with specific integration schemes, subclass the `Sampler` class and implement the `step` method tailored to your needs. The `step` method receives the current state `x_t`, `t`, and `t_next` from the base class and defines the integration scheme.

**Example: Euler Sampler**
```python
from rectified_flow.flow_components import Sampler

class EulerSampler(Sampler):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
    ):
        super().__init__(
            rectified_flow,
            num_steps,
            time_grid,
            record_traj_period,
            callbacks,
            num_samples,
        )

    def step(self, **model_kwargs):
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.rectified_flow.get_velocity(x_t, t, **model_kwargs)
        self.x_t = x_t + (t_next - t) * v_t
```
After defining your custom sampler, you can perform sampling just like with a standard sampler.

# Citation
If you find this repository useful for your research, please consider citing

```bibtex
@misc{lq2024rectifiedflow,
  author = {Qiang Liu, Runlong Liao, Bo Liu, Xixi Hu},
  title = {PyTorch RectifiedFlow},
  year = {2024},
  url = {https://github.com/lqiang67/rectified-flow}
}
```

# License

| Component | License                                                      |
| --------- | ------------------------------------------------------------ |
| Codebase  | [MIT License](LICENSE)                                       |

