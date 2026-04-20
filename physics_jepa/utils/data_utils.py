from einops import rearrange
import torch
from skimage.transform import resize
import numpy as np
from PIL import Image
import torch.nn.functional as F

def normalize_labels(x, stats={}):
    if 'mins' in stats and 'maxes' in stats:
        mins = torch.tensor(stats['mins'])
        maxes = torch.tensor(stats['maxes'])
        return (x - mins) / (maxes - mins)
    elif 'means' in stats and 'stds' in stats:
        if 'compression' in stats:
            for i, compression_type in enumerate(stats['compression']):
                if compression_type == 'log':
                    x[:, i] = torch.log10(x[:, i])
                elif compression_type == None:
                    continue
        means = torch.tensor(stats['means'])
        stds = torch.tensor(stats['stds'])
        return (x - means) / stds
    else:
        return x

def fft_resize_2d(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Band-limited FFT-based resize of the last two (spatial) axes of `x`.

    Assumes periodic boundary conditions. Works for up/down sampling and
    arbitrary even/odd target sizes. Short-circuits when input == output.
    Torch FFT does not support bf16/fp16, so low-precision inputs are
    cast to float32 internally and cast back on return.
    """
    in_h, in_w = x.shape[-2], x.shape[-1]
    if in_h == out_h and in_w == out_w:
        return x

    orig_dtype = x.dtype
    if orig_dtype not in (torch.float32, torch.float64):
        x = x.to(torch.float32)

    X = torch.fft.fft2(x, dim=(-2, -1))
    X = torch.fft.fftshift(X, dim=(-2, -1))

    # center crop or zero-pad to (out_h, out_w) around the DC bin
    def _center_range(n_in, n_out):
        # index range in the input that maps to the output, centered on DC
        start = (n_in - n_out) // 2
        return start, start + n_out

    # allocate target spectrum, then copy the overlapping central block
    out_shape = x.shape[:-2] + (out_h, out_w)
    Y = torch.zeros(out_shape, dtype=X.dtype, device=X.device)

    copy_h = min(in_h, out_h)
    copy_w = min(in_w, out_w)
    src_h0, src_h1 = _center_range(in_h, copy_h)
    src_w0, src_w1 = _center_range(in_w, copy_w)
    dst_h0, dst_h1 = _center_range(out_h, copy_h)
    dst_w0, dst_w1 = _center_range(out_w, copy_w)
    Y[..., dst_h0:dst_h1, dst_w0:dst_w1] = X[..., src_h0:src_h1, src_w0:src_w1]

    Y = torch.fft.ifftshift(Y, dim=(-2, -1))
    y = torch.fft.ifft2(Y, dim=(-2, -1)).real

    # preserve per-pixel amplitude under resampling
    y = y * (float(out_h * out_w) / float(in_h * in_w))

    if y.dtype != orig_dtype:
        y = y.to(orig_dtype)
    return y


def subsample(x, reso):
    """ Subsample a numpy array (or cpu tensor) to a given resolution 
    using antialiasing Gaussian filter. """
    ndim = len(reso)
    output_shape = x.shape[:-ndim] + tuple(reso)
    if any(output_shape[d] > x.shape[d] for d in range(-ndim, 0)):
        return x
    if output_shape == x.shape:
        return x
    if isinstance(x, np.ndarray):
        return torch.tensor(resize(x, output_shape, anti_aliasing=True))
    elif isinstance(x, torch.Tensor):
        x = x.numpy()
        x = resize(x, output_shape, anti_aliasing=True)
        return torch.tensor(x)

def mse(x, y):
    loss = (x - y).pow(2).mean()
    return {"loss": loss}

def mae(x, y):
    loss = (x - y).abs().mean()
    return {"loss": loss}