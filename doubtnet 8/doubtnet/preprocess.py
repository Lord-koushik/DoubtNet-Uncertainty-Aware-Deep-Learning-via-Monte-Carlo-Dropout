"""Turn an arbitrary user image (photo, scan, drawing) into an MNIST-style tensor."""
import numpy as np
import torch
from PIL import Image, ImageOps

from .data import MNIST_MEAN, MNIST_STD


def image_to_tensor(img: Image.Image, auto_invert: bool = True, center: bool = True) -> torch.Tensor:
    """Returns a normalised (1,1,28,28) tensor plus nothing else.

    MNIST digits are white strokes on black, roughly centred in a 20x20 box
    inside the 28x28 frame; we mimic that so the model sees familiar framing.
    """
    g = ImageOps.grayscale(img.convert("RGB"))
    a = np.asarray(g, dtype=np.float32) / 255.0
    if auto_invert and a.mean() > 0.5:          # dark ink on light paper -> invert
        a = 1.0 - a
    if center:
        mask = a > 0.15
        if mask.any():
            ys, xs = np.where(mask)
            a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = a.shape
        s = max(h, w)
        sq = np.zeros((s, s), dtype=np.float32)
        sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
        small = Image.fromarray((sq * 255).astype(np.uint8)).resize((20, 20), Image.LANCZOS)
        out = np.zeros((28, 28), dtype=np.float32)
        out[4:24, 4:24] = np.asarray(small, dtype=np.float32) / 255.0
        a = out
    else:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((28, 28)),
                       dtype=np.float32) / 255.0
    x = torch.from_numpy(a)[None, None]
    return (x - MNIST_MEAN) / MNIST_STD
