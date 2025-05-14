import os
import argparse
import math
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.inception import InceptionScore
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a PCA+KDE baseline on CIFAR-10")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Path to CIFAR-10 data directory.")
    parser.add_argument("--output_dir", type=str, default="./kde_output",
                        help="Where to save sampled images and metrics.")
    parser.add_argument("--pca_components", type=int, default=100,
                        help="Number of PCA components.")
    parser.add_argument("--bandwidth", type=float, default=0.1,
                        help="Bandwidth for Gaussian KDE.")
    parser.add_argument("--n_samples", type=int, default=50000,
                        help="Number of images to sample from KDE.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default=None,
                        help="Primary device for data processing (e.g., cuda, mps, cpu). If None, Accelerator decides.")
    return parser.parse_args()

class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted(Path(folder).glob("*.png"))
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def main():
    args = parse_args()
    accelerator = Accelerator()
    
    primary_device = torch.device(args.device if args.device else accelerator.device)
    metrics_device = torch.device("cpu") # Keep metrics on CPU for stability with dtypes
    
    set_seed(args.seed)
    print(f"Using primary device: {primary_device}")
    print(f"Using metrics device: {metrics_device}")

    # 1. Load CIFAR-10 as flattened arrays (for PCA/KDE fitting)
    transform_flat = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(), # Scales to [0,1] float
        transforms.Normalize((0.5,)*3, (0.5,)*3) # Normalizes to [-1,1] float
    ])
    cifar_train_for_kde = datasets.CIFAR10(root=args.data_root, train=True,
                             download=True, transform=transform_flat)
    # Data for PCA is float32, normalized to [-1, 1]
    data = torch.stack([img for img, _ in cifar_train_for_kde]).view(len(cifar_train_for_kde), -1).numpy().astype(np.float32)


    # 2. Fit PCA and KDE
    pca = PCA(n_components=args.pca_components)
    data_reduced = pca.fit_transform(data)
    kde = KernelDensity(kernel="gaussian", bandwidth=args.bandwidth)
    kde.fit(data_reduced)

    # 3. Sample and save images
    out_imgs = Path(args.output_dir) / "kde_samples"
    out_imgs.mkdir(parents=True, exist_ok=True)
    samples = kde.sample(args.n_samples).astype(np.float32)
    # recon will be in [-1, 1] range, float32
    recon = np.clip(pca.inverse_transform(samples), -1, 1)
    for i, vec in enumerate(tqdm(recon, desc="Saving KDE samples")):
        # Convert from [-1,1] float to [0,255] uint8 for saving as PNG
        img_to_save = ((vec.reshape(3,32,32).transpose(1,2,0) * 0.5 + 0.5) * 255).astype(np.uint8)
        Image.fromarray(img_to_save).save(out_imgs / f"img_{i:05d}.png")

    # 4. Prepare dataloaders for FID/IS
    # This transform produces float tensors in [-1, 1]
    normalize_eval = transforms.Normalize((0.5,)*3, (0.5,)*3)
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize_eval])
    
    pin_memory_flag = True if primary_device.type == 'cuda' else False

    # CIFAR10 dataset for "real" images for FID
    real_images_dataset = datasets.CIFAR10(root=args.data_root, train=True, # Using train split for FID real features
                                            download=True, transform=eval_tf) # Apply [-1,1] normalization
    real_loader = torch.utils.data.DataLoader(
        real_images_dataset,
        batch_size=256, shuffle=False, num_workers=4, pin_memory=pin_memory_flag)
    
    # Dataset for generated images
    gen_loader = torch.utils.data.DataLoader(
        ImageFolderDataset(out_imgs, transform=eval_tf), # Apply [-1,1] normalization
        batch_size=256, shuffle=False, num_workers=4, pin_memory=pin_memory_flag)

    # 5. Compute metrics
    fid = FID().to(metrics_device)
    iscore = InceptionScore().to(metrics_device) # InceptionScore also expects uint8

    # Function to convert [-1, 1] float tensors to [0, 255] uint8 tensors
    def denormalize_to_uint8(tensor_batch):
        # Input: float tensor batch, values in approx [-1, 1]
        # 1. Un-normalize to [0, 1]
        tensor_batch = tensor_batch * 0.5 + 0.5
        # 2. Clip to ensure values are strictly in [0, 1] after float arithmetic
        tensor_batch = torch.clamp(tensor_batch, 0.0, 1.0)
        # 3. Scale to [0, 255]
        tensor_batch = tensor_batch * 255.0
        # 4. Convert to uint8
        return tensor_batch.to(torch.uint8)

    # update real
    for images_real_normalized, _ in tqdm(real_loader, desc="Updating FID real"):
        images_real_uint8 = denormalize_to_uint8(images_real_normalized)
        fid.update(images_real_uint8.to(metrics_device), real=True)
        
    # update fake
    for images_fake_normalized in tqdm(gen_loader, desc="Updating FID & IS fake"):
        images_fake_uint8 = denormalize_to_uint8(images_fake_normalized)
        images_fake_uint8_metrics_device = images_fake_uint8.to(metrics_device)
        fid.update(images_fake_uint8_metrics_device, real=False)
        iscore.update(images_fake_uint8_metrics_device) # IS also needs uint8

    fid_score = fid.compute().item()
    is_mean, is_std = iscore.compute()
    is_mean = is_mean.item()
    is_std = is_std.item()

    # 6. Print and save
    print(f"KDE baseline: FID = {fid_score:.4f}, IS = {is_mean:.4f} ± {is_std:.4f}")
    with open(Path(args.output_dir)/"metrics.txt", "w") as f:
        f.write(f"FID {fid_score:.4f}\nIS_mean {is_mean:.4f}\nIS_std {is_std:.4f}\n")

if __name__ == "__main__":
    main()