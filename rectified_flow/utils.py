import torch
import random
import numpy as np
import torchvision
import matplotlib.pyplot as plt


def match_dim_with_data(
    t: torch.Tensor | float | list[float],
    X_shape: tuple,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    expand_dim: bool = True,
):
    """
    Prepares the time tensor by reshaping it to match the dimensions of X.

    Args:
        t (Union[torch.Tensor, float, List[float]]): Time tensor, which can be:
            - A scalar (float or 0-dimensional torch.Tensor)
            - A list of floats with length equal to the batch size or length 1
            - A torch.Tensor of shape (B,), (B, 1), or (1,)
        X_shape (tuple): Shape of the tensor X, e.g., X.shape

    Returns:
        torch.Tensor: Reshaped time tensor, ready for broadcasting with X.
    """
    B = X_shape[0]  # Batch size
    ndim = len(X_shape)

    if isinstance(t, float): # Create a tensor of shape (B,)
        t = torch.full((B,), t, device=device, dtype=dtype)
    elif isinstance(t, list):
        if len(t) == 1: # If t is a list of length 1, repeat the scalar value B times
            t = torch.full((B,), t[0], device=device, dtype=dtype)
        elif len(t) == B:
            t = torch.tensor(t, device=device, dtype=dtype)
        else:
            raise ValueError(f"Length of t list ({len(t)}) does not match batch size ({B}) and is not 1.")
    elif isinstance(t, torch.Tensor):
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 0: # Scalar tensor, expand to (B,)
            t = t.repeat(B)
        elif t.ndim == 1:
            if t.shape[0] == 1: # Tensor of shape (1,), repeat to (B,)
                t = t.repeat(B)
            elif t.shape[0] == B: # t is already of shape (B,)
                pass
            else:
                raise ValueError(f"Batch size of t ({t.shape[0]}) does not match X ({B}).")
        elif t.ndim == 2:
            if t.shape == (B, 1): # t is of shape (B, 1), squeeze last dimension
                t = t.squeeze(1)
            elif t.shape == (1, 1): # t is of shape (1, 1), expand to (B,)
                t = t.squeeze().repeat(B)
            else:
                raise ValueError(f"t must be of shape ({B}, 1) or (1, 1), but got {t.shape}")
        else:
            raise ValueError(f"t can have at most 2 dimensions, but got {t.ndim}")
    else:
        raise TypeError(f"t must be a torch.Tensor, float, or a list of floats, but got {type(t)}.")

    # Reshape t to have singleton dimensions matching X_shape after the batch dimension
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_cifar_results(images, nrow=10):
    images = (images.cpu().detach().clone() * 0.5 + 0.5).clamp(0, 1)
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=1, normalize=False)
    plt.figure(figsize=(10, (images.size(0) // nrow + 1) * 1.), dpi=300)
    np_grid = grid.permute(1, 2, 0).numpy()
    plt.imshow(np_grid)
    plt.axis('off')
    plt.show()
