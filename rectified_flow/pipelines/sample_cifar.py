import torch
import os
import argparse
from torchvision.utils import save_image
from scipy.integrate import solve_ivp
from rectified_flow.models.unet import SongUNet
from rectified_flow.models.dit import DiT

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.inference_mode()
def f_np(t, z, func, shape):
    z_tensor = torch.tensor(z, dtype=torch.float32, device=device).reshape(shape)
    t_tensor = t * torch.ones(shape[0], device=device)
    dz_tensor = func(z_tensor, t_tensor)
    return dz_tensor.detach().cpu().numpy().reshape((-1,))


@torch.inference_mode()
def rk45(f, z0, startT=0.0, endT=1.0):
    shape = z0.shape
    z0_np = z0.cpu().numpy().flatten()
    t_span = (startT, endT)
    sol = solve_ivp(
        f_np,
        t_span,
        z0_np,
        args=(f, shape),
        method="RK45",
        t_eval=None,
        rtol=1e-3,
        atol=1e-3,
    )
    z_final_np = sol.y[:, -1]
    nfe = sol.nfev
    print(f"Number of function evaluations: {nfe}")
    z_final = torch.tensor(z_final_np, dtype=torch.float32, device=device).reshape(
        shape
    )
    return z_final


def main(args):
    torch.manual_seed(args.seed)
    global device
    device = args.device

    if args.model_name == "unet":
        model_cls = SongUNet
    elif args.model_name == "dit":
        model_cls = DiT

    model = model_cls.from_pretrained(args.ckpt_path, use_ema=True).to(device)
    model.eval()

    print(f"Saving images to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.inference_mode():
        for iteration in range(args.iterations):
            print(
                f"Iteration: {iteration}, images from {iteration * args.batch_size} to {(iteration + 1) * args.batch_size}"
            )
            z_0 = torch.randn(args.batch_size, 3, 32, 32, device=device)
            z_1 = rk45(model, z_0, startT=0.0, endT=1.0)
            z_1 = (z_1 * 0.5 + 0.5).clamp(0, 1)
            for i in range(z_1.shape[0]):
                save_image(
                    z_1[i], os.path.join(args.output_dir, f"{iteration}_{i}.png")
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ODE-based image generation using DiT."
    )
    parser.add_argument(
        "--model_name", type=str, default="unet", help="Name of the model to use."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/root/autodl-tmp/unet_cifar/checkpoint-500000",
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/unet_cifar/checkpoint-500000/samples",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size for image generation."
    )
    parser.add_argument(
        "--iterations", type=int, default=500, help="Number of iterations to run."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    main(args)
