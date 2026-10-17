"""DoubtNet: a small CNN with dropout layers that stay active at test time.

Monte Carlo (MC) Dropout (Gal & Ghahramani, 2016) treats dropout as approximate
Bayesian inference: every forward pass with dropout switched on samples one
"thinned" network from an approximate posterior over weights. Averaging T such
passes gives a predictive distribution, and the spread between passes measures
how much the model doubts its answer.
"""
import torch
import torch.nn as nn


class DoubtNet(nn.Module):
    def __init__(self, num_classes: int = 10, p_conv: float = 0.25, p_fc: float = 0.5):
        super().__init__()
        self.p_conv, self.p_fc = p_conv, p_fc
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(p_conv),                     # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(p_conv),                     # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(inplace=True),
            nn.Dropout(p_fc),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """Put the model in eval mode but keep every Dropout layer sampling.

    (BatchNorm-style layers, if added later, stay in eval mode.)
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:                      # older torch without weights_only
        return torch.load(path, map_location=device)


def load_model(path: str, device: torch.device) -> DoubtNet:
    ckpt = _torch_load(path, device)
    cfg = ckpt.get("config", {})
    model = DoubtNet(p_conv=cfg.get("p_conv", 0.25), p_fc=cfg.get("p_fc", 0.5)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
