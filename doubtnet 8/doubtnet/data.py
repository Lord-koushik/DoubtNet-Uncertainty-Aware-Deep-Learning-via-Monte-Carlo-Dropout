"""In-distribution data (MNIST) and the out-of-distribution test suite.

OOD suite
  far-OOD      : Fashion-MNIST (clothes), KMNIST (Japanese characters)
  synthetic    : uniform noise, Gaussian noise, blank images
  near-OOD     : colour-inverted MNIST
  dataset shift: MNIST rotated 0..180 degrees (accuracy should fall, doubt should rise)
All images are 1x28x28 and normalised with MNIST statistics, exactly as a deployed
model would see them.
"""
import ssl
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

# macOS python.org builds ship without root certificates, which makes the dataset
# download fail with CERTIFICATE_VERIFY_FAILED. Use certifi's bundle when available.
try:
    import certifi

    def _certifi_ctx(*args, **kwargs):
        return ssl.create_default_context(cafile=certifi.where())

    ssl._create_default_https_context = _certifi_ctx
except Exception:
    pass

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
NORMALIZE = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
TO_TENSOR = transforms.ToTensor()
BASE_TF = transforms.Compose([TO_TENSOR, NORMALIZE])


def materialize(ds):
    """Run every PIL->tensor transform ONCE and keep the result in memory.
    Without this, each epoch / each MC pass re-decodes images one by one (the main slowdown)."""
    if isinstance(ds, TensorDataset):
        return ds
    dl = DataLoader(ds, batch_size=1000, shuffle=False, num_workers=0)
    xs, ys = zip(*[(x, y) for x, y in dl])
    return TensorDataset(torch.cat(xs), torch.cat(ys))


def mnist_train_val(root="data", val_size=5000, seed=0, augment=False):
    train_tf = BASE_TF
    if augment:
        train_tf = transforms.Compose([transforms.RandomAffine(8, translate=(0.08, 0.08)),
                                       TO_TENSOR, NORMALIZE])
    full_aug = datasets.MNIST(root, train=True, download=True, transform=train_tf)
    full_plain = datasets.MNIST(root, train=True, download=True, transform=BASE_TF)
    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(len(full_plain)),
                                      [len(full_plain) - val_size, val_size], generator=g)
    val = materialize(Subset(full_plain, list(val_idx)))
    train = Subset(full_aug, list(train_idx))
    if not augment:
        train = materialize(train)
    return train, val


def mnist_test(root="data"):
    return materialize(datasets.MNIST(root, train=False, download=True, transform=BASE_TF))


def _subset(ds, n, seed=0):
    if n is None or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    return Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())


def _synthetic(kind, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform_noise":
        x = torch.rand(n, 1, 28, 28, generator=g)
    elif kind == "gaussian_noise":
        x = (torch.randn(n, 1, 28, 28, generator=g) * 0.5 + 0.5).clamp(0, 1)
    elif kind == "blank":
        x = torch.zeros(n, 1, 28, 28)
    else:
        raise ValueError(kind)
    x = (x - MNIST_MEAN) / MNIST_STD
    return TensorDataset(x, torch.full((n,), -1, dtype=torch.long))


class _Mapped(torch.utils.data.Dataset):
    """Applies a tensor->tensor function to an MNIST-style dataset (pre-normalisation)."""

    def __init__(self, base, fn):
        self.base, self.fn = base, fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return NORMALIZE(self.fn(x)), y


def ood_datasets(root="data", n=2000, seed=0):
    """Dict name -> dataset. Labels are -1 when they don't mean an MNIST digit."""
    raw_mnist = datasets.MNIST(root, train=False, download=True, transform=TO_TENSOR)
    out = {
        "Fashion-MNIST": _subset(datasets.FashionMNIST(root, train=False, download=True,
                                                       transform=BASE_TF), n, seed),
        "KMNIST": _subset(datasets.KMNIST(root, train=False, download=True,
                                          transform=BASE_TF), n, seed),
        "Inverted MNIST": _subset(_Mapped(raw_mnist, lambda x: 1.0 - x), n, seed),
        "Uniform noise": _synthetic("uniform_noise", n, seed),
        "Gaussian noise": _synthetic("gaussian_noise", n, seed),
        "Blank": _synthetic("blank", min(n, 200), seed),
    }
    return {k: materialize(v) for k, v in out.items()}


def rotated_mnist(angle, root="data", n=2000, seed=0):
    raw = datasets.MNIST(root, train=False, download=True, transform=TO_TENSOR)
    return materialize(_subset(_Mapped(raw, lambda x: TF.rotate(x, float(angle))), n, seed))


def head(ds, n):
    """First n items of a dataset (used for the quick ablations)."""
    return ds if len(ds) <= n else Subset(ds, list(range(n)))


def loader(ds, batch_size=1024, shuffle=False, num_workers=0):
    # num_workers=0 is the safe default on Windows; raise it on Linux for speed.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def denormalize(x):
    return (x * MNIST_STD + MNIST_MEAN).clamp(0, 1)
