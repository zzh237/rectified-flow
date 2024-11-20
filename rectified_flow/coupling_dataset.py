import torch
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path
from functools import partial

class CouplingDataset(Dataset):
    def __init__(
        self, 
        D1: torch.utils.data.Dataset | torch.Tensor | dist.Distribution | callable,
        D0: torch.utils.data.Dataset | torch.Tensor | dist.Distribution | callable | None = None,
        reflow: bool = False,
    ):
        """
        A dataset that provides coupled samples from D0 and D1.

        Args:
            D1: Dataset, Tensor, distribution, or callable.
                The target data samples.
                All should return data after applying transforms, such as ToTensor().
            D0: None (default), Dataset, Tensor, distribution, or callable.
                The noise data samples. If None and reflow=False, defaults to standard normal.
            reflow: bool.
                If True, D0 and D1 are paired samples.
                If False, D0 is generated independently.
        """
        self.D1 = D1
        self.D0 = D0
        self.reflow = reflow

        self.D1_mode = self._get_D_mode(self.D1)
        
        if self.D0 is None and not self.reflow:
            self.D0 = self._set_default_noise()
            self.D0_mode = 'distribution'
        else:
            self.D0_mode = self._get_D_mode(self.D0)

        self.length = self._determine_length()

        self._validate_inputs()

    def _get_D_mode(self, D):
        if D is None:
            return 'none'
        elif isinstance(D, torch.Tensor):
            return 'tensor'
        elif isinstance(D, torch.utils.data.Dataset):
            return 'dataset'
        elif isinstance(D, torch.distributions.Distribution):
            return 'distribution'
        elif callable(D):
            return 'callable'
        else:
            raise NotImplementedError(f"Unsupported type: {type(D)}")

    def _set_default_noise(self):
        # Need to know sample shape
        sample_shape = self._get_sample_shape(self.D1)
        return dist.Normal(torch.zeros(sample_shape), torch.ones(sample_shape))

    def _get_sample_shape(self, D):
        sample = self._get_sample(D, 0, self._get_D_mode(D))
        return sample.shape

    def _determine_length(self):
        if self.reflow:
            len_D1 = self._get_length(self.D1)
            len_D0 = self._get_length(self.D0)
            assert len_D1 == len_D0, "D0 and D1 must have the same length when reflow=True"
            return len_D1
        else:
            return self._get_length(self.D1)

    def _get_length(self, D):
        if isinstance(D, torch.Tensor) or isinstance(D, torch.utils.data.Dataset):
            return len(D)
        else:
            return self.default_length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        X1 = self._get_sample(self.D1, index, self.D1_mode)
        if self.reflow:
            X0 = self._get_sample(self.D0, index, self.D0_mode)
        else:
            if self.D0_mode == 'distribution':
                X0 = None  # Will be handled in collate_fn
            else:
                X0 = self._get_sample(self.D0, index, self.D0_mode)
        return X0, X1

    def _get_sample(self, D, index, mode):
        if mode == 'tensor':
            return D[index]
        elif mode == 'dataset':
            sample = D[index]
            if isinstance(sample, tuple):
                sample = sample[0]
            return sample
        elif mode == 'distribution':
            # Return None, will be handled in collate_fn
            return None
        elif mode == 'callable':
            return D()
        elif mode == 'none':
            return None
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

    def _validate_inputs(self):
        if self.reflow:
            # D0 and D1 must be datasets or tensors of same length
            assert self.D0_mode in ['tensor', 'dataset'], "When reflow=True, D0 must be a tensor or dataset"
            assert self.D1_mode in ['tensor', 'dataset'], "When reflow=True, D1 must be a tensor or dataset"
            len_D0 = self._get_length(self.D0)
            len_D1 = self._get_length(self.D1)
            assert len_D0 == len_D1, "D0 and D1 must have the same length when reflow=True"
        else:
            # D1 must be provided
            assert self.D1 is not None, "D1 must be provided"

    default_length = 10000

# Custom collate function
def coupling_collate_fn(batch, D0_distribution=None):
    X0_list, X1_list = zip(*batch)
    # Check if X0_list contains all None
    if all(x is None for x in X0_list) and D0_distribution is not None:
        batch_size = len(X0_list)
        X0_batch = D0_distribution.sample([batch_size])
    else:
        X0_batch = torch.stack(X0_list)
    X1_batch = torch.stack(X1_list)
    return X0_batch, X1_batch

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
