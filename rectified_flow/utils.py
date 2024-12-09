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
            ".",
            label="D1",
            markersize=markersize,
            alpha=alpha_gt_points,
        )

    # Plot initial points from trajectories
    plt.plot(
        xtraj[0][:, dim0],
        xtraj[0][:, dim1],
        ".",
        label="D0",
        markersize=markersize,
        alpha=alpha_gt_points,
    )

    # Plot generated points
    plt.plot(
        xtraj[-1][:, dim0],
        xtraj[-1][:, dim1],
        "r.",
        label="Generated",
        markersize=markersize,
        alpha=alpha_generated_points,
    )

    # Plot trajectory lines
    for i in range(min(num_trajectories, xtraj.shape[1])):
        plt.plot(xtraj[:, i, dim0], xtraj[:, i, dim1], "--g", alpha=alpha_trajectories)

    # Add legend and adjust layout
    if show_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlabel(f"Dimension {dim0}")
    plt.ylabel(f"Dimension {dim1}")
    plt.title("2D Trajectories Visualization")
    plt.tight_layout()


def visualize_2d_trajectories_plotly(
    trajectories_dict: dict[str, list[torch.Tensor]],
    D1_gt_samples: torch.Tensor = None,
    num_trajectories: int = 50,
    markersize: int = 3,
    dimensions: list[int] = [0, 1],
    alpha_trajectories: float = 0.5,
    alpha_particles: float = 0.8,
    alpha_gt_points: float = 1.0,
    show_legend: bool = True,
    title: str = "2D Trajectories Visualization",
):
    import plotly.graph_objects as go

    dim0, dim1 = dimensions

    # Convert ground truth samples to NumPy if provided
    if D1_gt_samples is not None:
        D1_gt_samples = D1_gt_samples.clone().to(torch.float32).cpu().detach().numpy()

    # Prepare color mapping for trajectories
    particle_colors = [
        "#1E90FF",
        "#FF69B4",
        "#7B68EE",
        "#FF8C00",
        "#32CD32",
        "#4169E1",
        "#FF4500",
        "#9932CC",
        "#ADFF2F",
        "#FFD700",
    ]

    trajectory_colors = [
        "#8ABDE5",
        "#E09CAF",
        "#B494E1",
        "#E5B680",
        "#82C9A1",
        "#92BCD5",
        "#E68FA2",
        "#A98FC8",
        "#E5C389",
        "#A0C696",
    ]

    trajectory_names = list(trajectories_dict.keys())
    colors = {}
    for i, name in enumerate(trajectory_names):
        colors[name] = {
            "particle_color": particle_colors[i % len(particle_colors)],
            "trajectory_color": trajectory_colors[i % len(trajectory_colors)],
        }

    # Process trajectories and store data
    trajectory_data = {}
    max_time_steps = 0

    for trajectory_name, traj_list in trajectories_dict.items():
        # traj_list: List of trajectories, each of shape [batch_size, dimension]
        xtraj_list = [
            traj.clone().to(torch.float32).detach().cpu().numpy() for traj in traj_list
        ]
        xtraj = np.stack(xtraj_list)  # Shape: [time_steps, total_batch_size, dimension]
        trajectory_data[trajectory_name] = xtraj
        max_time_steps = max(max_time_steps, xtraj.shape[0])

    # Create figure
    fig = go.Figure()

    # Plot ground truth samples
    if D1_gt_samples is not None:
        fig.add_trace(
            go.Scatter(
                x=D1_gt_samples[:, dim0],
                y=D1_gt_samples[:, dim1],
                mode="markers",
                name="D1 ground truth",
                marker=dict(size=markersize, opacity=alpha_gt_points, color="red"),
                showlegend=show_legend,
            )
        )

    particle_trace_indices = {}
    particle_traces_info = []
    current_trace_index = len(fig.data)

    for trajectory_name, xtraj in trajectory_data.items():
        particle_color = colors[trajectory_name]["particle_color"]
        trajectory_color = colors[trajectory_name]["trajectory_color"]
        time_steps = xtraj.shape[0]
        num_points = xtraj.shape[1]
        indices = np.arange(min(num_trajectories, num_points))

        # Plot all trajectories as single trace
        all_line_x = []
        all_line_y = []

        for i in indices:
            line_x = xtraj[:, i, dim0]
            line_y = xtraj[:, i, dim1]
            all_line_x.extend(line_x.tolist() + [np.nan])
            all_line_y.extend(line_y.tolist() + [np.nan])

        fig.add_trace(
            go.Scatter(
                x=all_line_x,
                y=all_line_y,
                mode="lines",
                name=f"{trajectory_name}",
                line=dict(dash="solid", color=trajectory_color, width=1.5),
                opacity=alpha_trajectories,
                showlegend=True,
            )
        )
        current_trace_index += 1

        # Plot initial points from trajectories
        fig.add_trace(
            go.Scatter(
                x=xtraj[0, :, dim0],
                y=xtraj[0, :, dim1],
                mode="markers",
                name=f"{trajectory_name} x_0",
                marker=dict(size=markersize, opacity=alpha_gt_points, color="blue"),
                showlegend=False,
            )
        )
        current_trace_index += 1

        # Collect particle traces info
        particle_traces_info.append(
            {
                "trajectory_name": trajectory_name,
                "x": xtraj[0, :, dim0],
                "y": xtraj[0, :, dim1],
                "particle_color": particle_color,
                "trace_index": None,
            }
        )

    # Plot after all trajectories
    for info in particle_traces_info:
        fig.add_trace(
            go.Scatter(
                x=info["x"],
                y=info["y"],
                mode="markers",
                name=f"{info['trajectory_name']}",
                marker=dict(size=markersize, color=info["particle_color"]),
                showlegend=False,
            )
        )
        info["trace_index"] = current_trace_index
        particle_trace_indices[info["trajectory_name"]] = current_trace_index
        current_trace_index += 1

    # Create frames
    frames = []
    for t in range(max_time_steps):
        frame_data = []
        frame_trace_indices = []
        for trajectory_name, xtraj in trajectory_data.items():
            particle_color = colors[trajectory_name]["particle_color"]

            if t >= xtraj.shape[0]:
                continue

            x = xtraj[t, :, dim0]
            y = xtraj[t, :, dim1]
            trace_index = particle_trace_indices[trajectory_name]

            frame_data.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=markersize, color=particle_color),
                    opacity=alpha_particles,
                    name=f"{trajectory_name} x_t",
                    showlegend=True,
                )
            )
            frame_trace_indices.append(trace_index)

        frames.append(
            go.Frame(data=frame_data, name=str(t), traces=frame_trace_indices)
        )

    # Create slider steps
    slider_steps = []
    for t in range(max_time_steps):
        step = dict(
            method="animate",
            args=[
                [str(t)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
            label=str(t),
        )
        slider_steps.append(step)

    # Create sliders
    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Step: "},
            pad={"t": 50},
            steps=slider_steps,
        )
    ]

    # Update figure layout
    fig.update_layout(
        sliders=sliders,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=500, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    )
                ],
            )
        ],
        xaxis_title=f"Dimension {dim0}",
        yaxis_title=f"Dimension {dim1}",
        title=title,
        showlegend=show_legend,
        height=600,
        width=900,
    )

    # Add frames
    fig.frames = frames

    # Show figure
    fig.show()


def plot_cifar_results(images, nrow=10, title=None):
    images = (images.cpu().detach().clone() * 0.5 + 0.5).clamp(0, 1)
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=1, normalize=False)
    plt.figure(figsize=(10, (images.size(0) // nrow + 1) * 1.0), dpi=300)
    np_grid = grid.permute(1, 2, 0).numpy()
    plt.imshow(np_grid)
    plt.axis("off")
    if title is not None:
        plt.title(title, fontsize=16)
    plt.show()
