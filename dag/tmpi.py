import numpy as np
import scipy.linalg as slin
import torch


from .tmpi_torch import h_tmpi_native_double, h_tmpi_native_float, h_fast_geo_float, h_fast_geo_double


class HTorchTmpi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, eps=1e-6):
        if B.dtype == torch.float64:
            h, G_h = h_tmpi_native_double(B, eps)
        elif B.dtype == torch.float32:
            h, G_h = h_tmpi_native_float(B, eps)
        ctx.save_for_backward(G_h)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        G_h, = ctx.saved_tensors
        return grad_output * G_h, None



IplusB = None
fminusI = None
T3 = None
eye = None

old_g = None
old_B = None
sec_g = None


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def h_fast_tmpi(B, eps=1e-6):
    """
    - Use binary search and recursion to compute a crude estimate of h_tmpi.
    - The output may be very differnt from h_tmpi and h_fast_tmpi_1.
    - Requires O(log k) matrix multiplications.
    - Different from h_fast_tmpi_1, here we only perform the first binary search.
    - When we are searching from j = 1, 2, 4, 8, 16, ..., if we found a power of B
      that satisfies np.max(np.abs(B)) <= eps, then we simply return it. If not, we
      further compute and return h_d(B), which is identical to the value of h_fast_geometric.
    - E.g., if k is 25, then return h_32(B).
    - E.g., if k is 41, then return h_64(B).
    - If no such k is found, i.e., np.max(np.abs(B)) <= eps is not satisfied for all
      power from 1 to d, then we return h_d(B).
    - Therefore, this algorithm is usually faster than h_fast_tmpi_1.
    - This algorithm is almost identical to h_fast_geometric, except that it ends earlier
      during the binary search. So it should be faster than h_fast_geometric in most cases.
    """
    d = B.shape[0]
    global old_g
    global old_B
    global sec_g
    if old_g is None:
        old_g = np.zeros_like(B)
        old_B = np.zeros_like(B)
        sec_g = np.zeros_like(B)
    if old_g.shape[0] != d:
        old_g = np.zeros_like(B)
        old_B = np.zeros_like(B)
        sec_g = np.zeros_like(B)
    _B = np.copy(B)
    _g = np.copy(B)
    _grad = np.eye(d)

    j = 1
    max_d = next_power_of_2(d)
    while 2 * j <= max_d:
        np.copyto(old_B, _B)
        np.copyto(old_g, _g)

        np.matmul(_B, _g, out=_g)
        np.add(old_g, _g, out=_g)

        np.copyto(sec_g, _grad)
        np.matmul(_B.T, _grad, out=_grad)
        np.matmul(_B, _B, out=_B)
        np.add(_grad, sec_g, out=_grad)
        np.copyto(sec_g, _g)

        sec_g -= old_g
        sec_g += old_B
        sec_g -= _B
        sec_g *= j

        _grad += sec_g.T

        if np.max(np.abs(_B)) < eps:
            break
        j *= 2

    return np.trace(_g), _grad
