import numpy as np
import scipy.linalg as slin
import torch

def h_exponential_torch(B):
    d = B.shape[0]
    h = torch.trace(torch.matrix_exp(B)) - d
    return h


def h_binomial_torch(B):
    d = B.shape[0]
    h = torch.trace(
        torch.matrix_power(torch.eye(d).to(B.device) + 1.0 / d * B, d)) - d
    return h


def h_exponential(B: np.ndarray):
    """h = tr(e^B) - d."""
    d = len(B)
    E = slin.expm(B)  # (Zheng et al. 2018)
    h = np.trace(E) - d
    G_h = E.T
    return h, G_h



def h_binomial(B):
    """
    - Require O(log d) matrix multiplications.
    - h = tr(I + B / d)^d - d.
    """
    d = len(B)
    M = np.eye(d) + B / d  # (Yu et al. 2019)
    E = np.linalg.matrix_power(M, d - 1)
    h = (E.T * M).sum() - d
    G_h = E.T
    return h, G_h

