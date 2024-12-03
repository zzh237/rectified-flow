import torch
import random
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def match_dim_with_data(
    t: torch.Tensor | float | list[float],
    x_shape: tuple,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    expand_dim: bool = True,
):
    r"""
    Format the time tensor `t` to match the batch size and dimensions of the data.

    This function ensures that the time tensor `t` is properly formatted to match the batch size specified by `x_shape`. 
    It handles various input types for `t`, including scalars, lists, or tensors, and converts `t` into a tensor with 
    appropriate shape, device, and dtype. Optionally, it can expand `t` to match the data dimensions beyond the batch size.

    Args:
        t (`torch.Tensor`, `float`, or `list[float]`): 
            The time(s) to be matched with the data dimensions. Can be a scalar, a list of floats, or a tensor.
        x_shape (`tuple`): 
            The shape of the data tensor, typically `(batch_size, ...)`.
        device (`torch.device`, optional, defaults to `torch.device("cpu")`): 
            The device on which to place the time tensor.
        dtype (`torch.dtype`, optional, defaults to `torch.float32`): 
            The data type of the time tensor.
        expand_dim (`bool`, optional, defaults to `True`): 
            Whether to expand `t` to match the dimensions after the batch dimension.

    Returns:
        t_reshaped (`torch.Tensor`): 
            The time tensor `t`, formatted to match the batch size or dimensions of the data.

    Example:
        ```python
        >>> x_shape = (16, 3, 32, 32)
        >>> t_prepared = match_dim_with_data([0.5], x_shape, expand_dim=True)
        >>> t_prepared.shape
        torch.Size([16, 1, 1, 1])

        >>> x_shape = (16, 3, 32, 32)
        >>> t_prepared = match_dim_with_data([0.5], x_shape, expand_dim=False)
        >>> t_prepared.shape
        torch.Size([16])
        ```
    """
    B = x_shape[0]  # Batch size
    ndim = len(x_shape)

    if isinstance(t, float):
        # Create a tensor of shape (B,) with the scalar value
        t = torch.full((B,), t, device=device, dtype=dtype)
    elif isinstance(t, list):
        if len(t) == 1:
            # If t is a list of length 1, repeat the scalar value B times
            t = torch.full((B,), t[0], device=device, dtype=dtype)
        elif len(t) == B:
            t = torch.tensor(t, device=device, dtype=dtype)
        else:
            raise ValueError(
                f"Length of t list ({len(t)}) does not match batch size ({B}) and is not 1."
            )
    elif isinstance(t, torch.Tensor):
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 0:
            # Scalar tensor, expand to (B,)
            t = t.repeat(B)
        elif t.ndim == 1:
            if t.shape[0] == 1:
                # Tensor of shape (1,), repeat to (B,)
                t = t.repeat(B)
            elif t.shape[0] == B:
                # t is already of shape (B,)
                pass
            else:
                raise ValueError(
                    f"Batch size of t ({t.shape[0]}) does not match x ({B})."
                )
        elif t.ndim == 2:
            if t.shape == (B, 1):
                # t is of shape (B, 1), squeeze last dimension
                t = t.squeeze(1)
            elif t.shape == (1, 1):
                # t is of shape (1, 1), expand to (B,)
                t = t.squeeze().repeat(B)
            else:
                raise ValueError(
                    f"t must be of shape ({B}, 1) or (1, 1), but got {t.shape}"
                )
        else:
            raise ValueError(f"t can have at most 2 dimensions, but got {t.ndim}")
    else:
        raise TypeError(
            f"t must be a torch.Tensor, float, or a list of floats, but got {type(t)}."
        )

    # Reshape t to have singleton dimensions matching x_shape after the batch dimension
    if expand_dim:
        expanded_dims = [1] * (ndim - 1)
        t = t.view(B, *expanded_dims)

    return t

    

def visualize_2d_trajectories(
    trajectories_list: list[torch.Tensor],
    D1_gt_samples: torch.Tensor = None,
    num_trajectories: int = 50,
    markersize: int = 3,
    dimensions: list[int] = [0, 1],
    alpha_trajectories: float = 0.5,
    alpha_generated_points: float = 1.0,
    alpha_gt_points: float = 1.0,
    show_legend: bool = True,
):
    """
    Plots 2D trajectories and points for visualization.

    Parameters:
        trajectories_list (list): List of trajectories to display.
        D1_gt_samples (torch.Tensor, optional): Ground truth samples.
        num_trajectories (int): Number of trajectories to display.
        markersize (int): Size of the markers.
        dimensions (list): Indices of the dimensions to plot.
        alpha_trajectories (float): Transparency of trajectory lines.
        alpha_generated_points (float): Transparency of generated points.
        alpha_gt_points (float): Transparency of true points.
    """
    dim0, dim1 = dimensions

    # Convert ground truth samples to NumPy if provided
    if D1_gt_samples is not None:
        D1_gt_samples = D1_gt_samples.clone().to(torch.float32).cpu().detach().numpy()

    # Flatten and stack trajectories, then convert to NumPy
    traj_list_flat = [
        traj.clone().to(torch.float32).detach().cpu().reshape(traj.shape[0], -1)
        for traj in trajectories_list
    ]

    xtraj = torch.stack(traj_list_flat).numpy()

    # Plot ground truth samples
    if D1_gt_samples is not None:
        plt.plot(
            D1_gt_samples[:, dim0],
            D1_gt_samples[:, dim1],
            '.',
            label='D1',
            markersize=markersize,
            alpha=alpha_gt_points
        )

    # Plot initial points from trajectories
    plt.plot(
        xtraj[0][:, dim0],
        xtraj[0][:, dim1],
        '.',
        label='D0',
        markersize=markersize,
        alpha=alpha_gt_points
    )

    # Plot generated points
    plt.plot(
        xtraj[-1][:, dim0],
        xtraj[-1][:, dim1],
        'r.',
        label='Generated',
        markersize=markersize,
        alpha=alpha_generated_points
    )

    # Plot trajectory lines
    for i in range(min(num_trajectories, xtraj.shape[1])):
        plt.plot(
            xtraj[:, i, dim0],
            xtraj[:, i, dim1],
            '--g',
            alpha=alpha_trajectories
        )

    # Add legend and adjust layout
    if show_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel(f'Dimension {dim0}')
    plt.ylabel(f'Dimension {dim1}')
    plt.title('2D Trajectories Visualization')
    plt.tight_layout()


from ipywidgets import interact, IntSlider

def visualize_2d_trajectories_interactive(
    trajectories_list: list[torch.Tensor],
    D1_gt_samples: torch.Tensor = None,
    num_trajectories: int = 50,
    markersize: int = 3,
    dimensions: list[int] = [0, 1],
    alpha_trajectories: float = 0.5,
    alpha_generated_points: float = 1.0,
    alpha_gt_points: float = 1.0,
    show_legend: bool = True,
):
    """
    Plots 2D trajectories and points for visualization with an interactive slider for time t.

    Parameters:
        trajectories_list (list): List of trajectories to display.
        D1_gt_samples (torch.Tensor, optional): Ground truth samples.
        num_trajectories (int): Number of trajectories to display.
        markersize (int): Size of the markers.
        dimensions (list): Indices of the dimensions to plot.
        alpha_trajectories (float): Transparency of trajectory lines.
        alpha_generated_points (float): Transparency of generated points.
        alpha_gt_points (float): Transparency of true points.
        show_legend (bool): Whether to display the legend.
    """
    dim0, dim1 = dimensions

    # Convert ground truth samples to NumPy if provided
    if D1_gt_samples is not None:
        D1_gt_samples = D1_gt_samples.clone().to(torch.float32).cpu().detach().numpy()

    # Concatenate trajectories along batch dimension
    traj_list_flat = [
        traj.clone().to(torch.float32).detach().cpu()  # Shape: [time_steps, batch_size, dimension]
        for traj in trajectories_list
    ]

    xtraj = torch.stack(traj_list_flat).numpy()  # Shape: [time_steps, total_batch_size, dimension]

    time_steps = xtraj.shape[0]

    # Precompute the trajectory lines
    lines_data = []
    num_points = xtraj.shape[1]
    for i in range(min(num_trajectories, num_points)):
        line_x = xtraj[:, i, dim0]
        line_y = xtraj[:, i, dim1]
        lines_data.append((line_x, line_y))

    # Define the plotting function
    def plot_at_t(t):
        fig, ax = plt.subplots(dpi=100)

        # Plot ground truth samples
        if D1_gt_samples is not None:
            ax.plot(
                D1_gt_samples[:, dim0],
                D1_gt_samples[:, dim1],
                '.',
                label='D1',
                markersize=markersize,
                alpha=alpha_gt_points
            )

        # Plot initial points from trajectories (at t=0)
        ax.plot(
            xtraj[0][:, dim0],
            xtraj[0][:, dim1],
            '.',
            label='D0',
            markersize=markersize,
            alpha=alpha_gt_points
        )

        # Plot generated points at final time
        ax.plot(
            xtraj[-1][:, dim0],
            xtraj[-1][:, dim1],
            'r.',
            label='Generated',
            markersize=markersize,
            alpha=alpha_generated_points
        )

        # Plot trajectory lines for num_trajectories
        for line_x, line_y in lines_data:
            ax.plot(
                line_x,
                line_y,
                '--g',
                alpha=alpha_trajectories
            )

        # Plot points at time t
        t_int = int(t)
        ax.plot(
            xtraj[t_int][:, dim0],
            xtraj[t_int][:, dim1],
            'o',
            label=f'step = {t_int}',
            markersize=markersize,
            color='blue'
        )

        # Add legend and labels
        if show_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel(f'Dimension {dim0}')
        ax.set_ylabel(f'Dimension {dim1}')
        ax.set_title(f'2D Trajectories Visualization at step={t_int}')
        plt.show()

    # Create the interactive slider
    interact(plot_at_t, t=IntSlider(min=0, max=time_steps - 1, step=1, value=0))


def plot_cifar_results(images, nrow=10, title=None):
    images = (images.cpu().detach().clone() * 0.5 + 0.5).clamp(0, 1)
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=1, normalize=False)
    plt.figure(figsize=(10, (images.size(0) // nrow + 1) * 1.), dpi=300)
    np_grid = grid.permute(1, 2, 0).numpy()
    plt.imshow(np_grid)
    plt.axis('off')
    if title is not None:
        plt.title(title, fontsize=16) 
    plt.show()
