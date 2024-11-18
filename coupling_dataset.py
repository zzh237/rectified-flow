import torch
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader

class CouplingDataset(Dataset):
    def __init__(self, noise=None, data=None, labels=None, independent_coupling=True):
        """
        Initialize the dataset with noise (D0), data (D1), and optional labels.

        Args:
            Uncoupled dataset:
                noise: Distribution or None. If None, defaults to standard normal distribution.
                data: Tensor, distribution, or callable for data samples.
                labels: Optional tensor for labels.
                independent_coupling: If True, D0 and D1 are sampled independently.
            Coupled dataset:
                noise: Tensor
                data: Tensor
                labels: Optional tensor for labels.
                independent_coupling: If False, D0 and D1 are paired samples.
        """
        self.D1 = data
        self.D0 = noise if noise is not None else self._set_default_noise()
        self.labels = labels
        self.independent_coupling = independent_coupling
        self.paired = not independent_coupling
        self._varlidate_inputs()
        self.default_dataset_length = 10000

    def _get_D_mode(self, D):
        if D is None:
            return 'default_noise'
        elif isinstance(D, torch.Tensor):
            return 'tensor'
        elif isinstance(D, torch.distributions.Distribution):
            return 'distribution'
        elif callable(D):
            return 'callable'
        else:
            raise NotImplementedError(f"Unsupported type: {type(D)}")

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
